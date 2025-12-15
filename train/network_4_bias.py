import sys, os
sys.path.append("..")

import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import nibabel as nib

import torch
from torch import nn
import torch.nn.functional as F

from dataloaders.flash_dataset import FlashMRIDataset
from models.avantika.unet import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data/processed"
PARAM_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/ground_truth_maps_processed"
SAVE_DIR   = "../redoing_stuff_12_14/network4_bias"

# ------------------------------------------------
# LS PARAMETER MAP LOADER
# ------------------------------------------------
def load_ls_params(subj_id):
    """
    Load least-squares parameter maps for one subject.

    Assumes:
        T2*:   sub-XXXXX_T2star.npy   -> (H, W, D)
        T1rho: sub-XXXXX_T1p.npy -> (H, W, D)

    Returns:
        params_np: np.ndarray of shape (2, H, W, D)
                   index 0 -> T2*
                   index 1 -> T1rho
    """
    t2s_path = os.path.join(PARAM_DIR, f"{subj_id}_T2star_ls.npy")
    t1r_path = os.path.join(PARAM_DIR, f"{subj_id}_T1rho_ls.npy")

    if not os.path.exists(t2s_path):
        raise FileNotFoundError(f"Missing LS T2* map: {t2s_path}")
    if not os.path.exists(t1r_path):
        raise FileNotFoundError(f"Missing LS T1rho map: {t1r_path}")

    t2s   = np.load(t2s_path)   # (H, W, D)
    t1rho = np.load(t1r_path)   # (H, W, D)

    # stack as (2, H, W, D)
    params = np.stack([t2s, t1rho], axis=0)
    return params


# ------------------------------------------------
# SUPERVISED DATASET
# ------------------------------------------------
class FlashMRIDatasetSupervised(torch.utils.data.Dataset):
    """
    Supervised wrapper for Network 2.

    Inputs:
      - multi-echo FLASH slices from DATA_DIR
      - least-squares parameter maps from PARAM_DIR

    Returns (per slice):
        x    : (C, H, W)   multi-echo input
        p_gt : (2, H, W)   [T2*, T1rho] ground-truth maps
    """
    def __init__(self, subject_ids, data_dir, param_dir, echo_indices=None):
        self.subject_ids = subject_ids
        self.data_dir = data_dir
        self.param_dir = param_dir
        self.echo_indices = echo_indices or [1, 2, 3, 4]

        # build (subject, slice_idx) index
        self.index = []
        for subj_id in subject_ids:
            first_echo_path = os.path.join(data_dir, f"{subj_id}_echo{self.echo_indices[0]}.npy")
            if not os.path.exists(first_echo_path):
                raise FileNotFoundError(f"Missing echo file for {subj_id}: {first_echo_path}")
            depth = np.load(first_echo_path, mmap_mode="r").shape[2]
            for z in range(depth):
                self.index.append((subj_id, z))

        print(f"[SUPERVISED] dataset built with {len(self.index)} slices "
              f"from {len(subject_ids)} subjects.")

        # small caches to avoid reloading from disk
        self._volume_cache = {}
        self._param_cache = {}

    def __len__(self):
        return len(self.index)

    def _load_subject_inputs(self, subj_id):
        vols = []
        for echo in self.echo_indices:
            path = os.path.join(self.data_dir, f"{subj_id}_echo{echo}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing echo file: {path}")
            vol = np.load(path)  # (H, W, D)
            vols.append(vol)
        stacked = np.stack(vols, axis=-1)  # (H, W, D, N_echoes)
        return stacked

    def _load_subject_params(self, subj_id):
        if subj_id in self._param_cache:
            return self._param_cache[subj_id]
        params = load_ls_params(subj_id)  # (2, H, W, D)
        self._param_cache[subj_id] = params
        return params

    def __getitem__(self, idx):
        subj_id, z = self.index[idx]

        # multi-echo input volume
        if subj_id in self._volume_cache:
            vol = self._volume_cache[subj_id]
        else:
            vol = self._load_subject_inputs(subj_id)
            self._volume_cache[subj_id] = vol

        # LS parameter maps
        params = self._load_subject_params(subj_id)

        # slice along depth dimension
        slice_img   = vol[:, :, z, :]      # (H, W, N_echoes)
        slice_param = params[:, :, :, z]   # (2, H, W)

        x    = torch.from_numpy(slice_img).permute(2, 0, 1).float()  # (C, H, W)
        p_gt = torch.from_numpy(slice_param).float()                 # (2, H, W)

        return x, p_gt

subjects = sorted(set([
    os.path.basename(f).split("_echo")[0]
    for f in glob.glob(os.path.join(DATA_DIR, "sub-*_echo1.npy"))
]))
print(f"Found {len(subjects)} subjects total.")

# drop known bad
subjects = [s for s in subjects if s != "sub-04620"]
print(f"Remaining {len(subjects)} subjects after removing corrupted entries.")

train_subj, val_subj = train_test_split(subjects, test_size=0.125, random_state=42)
print(f"Training on {len(train_subj)} subjects, validating on {len(val_subj)} subjects.")

TRAIN_SUBJECTS = train_subj
VAL_SUBJECTS   = val_subj

# All echoes present
ECHO_INDICES_ALL = [1, 2, 3, 4]
N_ECHOES = len(ECHO_INDICES_ALL)

N_PARAMS   = 2   # [T2*, T1rho]
BATCH_SIZE = 4
NUM_EPOCHS = 100
LR         = 1e-4
os.makedirs(SAVE_DIR, exist_ok=True)

# Echo times in seconds (aligned with ECHO_INDICES_ALL)
TEs_all = torch.tensor([0.012, 0.028, 0.044, 0.060], device=device)  # (4,)

# ------------------------------------------------
# PHYSICS MODEL
# ------------------------------------------------
def flash_forward_single_te(params, te_vals):
    """
    FLASH forward model for a single TE per sample.

    params: (B, 2, H, W)  -> [T2*, T1rho]
    te_vals: (B,)  scalar TE per sample (seconds)

    Returns:
        y_hat: (B, 1, H, W)
    """
    T2s   = torch.abs(params[:, 0:1, :, :]) + 1e-3
    T1rho = torch.abs(params[:, 1:2, :, :]) + 1e-3

    # reshape TEs to broadcast
    te = te_vals.view(-1, 1, 1, 1)   # (B,1,1,1)
    y_hat = T1rho * torch.exp(-te / T2s)  # (B,1,H,W)
    return y_hat

def nmse(pred, target, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalized MSE between pred and target.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
    pred   = pred.float()
    target = target.float()
    B      = pred.shape[0]
    pred_f = pred.view(B, -1)
    targ_f = target.view(B, -1)
    diff   = pred_f - targ_f
    num    = (diff * diff).sum(dim=1)
    denom  = (targ_f * targ_f).sum(dim=1)
    nmse_per_sample = num / (denom + eps)
    return nmse_per_sample.mean()

def safe_masked_mse(y_hat, y, mask):
    # y_hat, y: (B,1,H,W)
    # mask: (B,H,W) bool or 0/1
    mask = mask.unsqueeze(1).float()  # (B,1,H,W)
    diff2 = (y_hat - y)**2 * mask
    denom = mask.sum()
    if denom.item() == 0:
        return torch.tensor(0.0, device=y_hat.device)
    return diff2.sum() / denom

# ------------------------------------------------
# DATASET / DATALOADERS
# ------------------------------------------------
train_ds = FlashMRIDataset(TRAIN_SUBJECTS, DATA_DIR,
                           echo_indices=ECHO_INDICES_ALL,
                           mode="train")
val_ds   = FlashMRIDataset(VAL_SUBJECTS,   DATA_DIR,
                           echo_indices=ECHO_INDICES_ALL,
                           mode="val")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels=None, kernel_size=3):
        out_channels = out_channels or mid_channels
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU(inplace=False),
        )
    def forward_skip(self, x):
        second_to_last = None
        for layer in self:
            second_to_last = x
            x = layer(x)
        return x, second_to_last


class UNet_TEBias(nn.Module):
    def __init__(self, in_channels, out_channels, te_emb_dim=16):
        super().__init__()

        self.maxpool  = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ---------- Encoder ----------
        self.encoder = nn.ModuleList([
            ConvBlock(in_channels, 8),
            ConvBlock(8, 16),
            ConvBlock(16, 32),
        ])

        # ---------- Bottleneck ----------
        self.bottleneck = ConvBlock(32, 64, 32)

        # ---------- Decoder ----------
        self.decoder = nn.ModuleList([
            ConvBlock(64, 16),
            ConvBlock(32, 8),
            ConvBlock(16, 8),
        ])

        self.conv_last = nn.Conv2d(8, out_channels, kernel_size=1)

        # ---------- TE → bias ----------
        self.te_bias = nn.Sequential(
            nn.Linear(1, te_emb_dim),
            nn.ReLU(),
            nn.Linear(te_emb_dim, 32)   # must match bottleneck output channels
        )

    def forward(self, x, te_vals):
        """
        x: (B,1,H,W)
        te_vals: (B,1)
        """
        skips = []

        # Encoder
        for block in self.encoder:
            x, skip = block.forward_skip(x)
            x = self.maxpool(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # ---- TE bias injection ----
        b = self.te_bias(te_vals).view(x.size(0), x.size(1), 1, 1)
        x = x + b

        # Decoder
        for block, skip in zip(self.decoder, reversed(skips)):
            x = self.upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return self.conv_last(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet_TEBias(in_channels=1, out_channels=2).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                 factor=0.5, patience=5)

train_losses = []
val_losses = []
best_val_loss = float("inf")

# ------------------------------------------------
# TRAINING LOOP (self-supervised, brain-masked, FiLM)
# ------------------------------------------------
import torch
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.measure import label

MASK_PERCENTILE = 60.0

def build_brain_mask_batch(echo_img_batch):
    """
    Build per-slice brain masks from a batch of echo images.
    
    echo_img_batch: (B,1,H,W)
    Returns: masks of shape (B,H,W) boolean tensor
    """
    masks = []
    B, _, H, W = echo_img_batch.shape
    for b in range(B):
        echo = echo_img_batch[b,0].cpu().numpy()
        thr = np.percentile(echo[echo>0], MASK_PERCENTILE) if np.any(echo>0) else 0
        mask = echo > thr
        mask = binary_closing(mask, structure=np.ones((3,3)))
        mask = binary_fill_holes(mask)
        label_map = label(mask)
        if label_map.max() > 0:
            largest_cc = np.argmax(np.bincount(label_map.flat)[1:]) + 1
            mask = (label_map == largest_cc)
        masks.append(torch.from_numpy(mask))
    masks = torch.stack(masks).to(device)  # (B,H,W)
    return masks

# ------------------------------------------------
# TRAINING
# ------------------------------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for x_full, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        x_full = x_full.to(device)  # (B, N_echoes, H, W)
        B, N_echoes, H, W = x_full.shape

        # ---- randomly pick one echo per sample ----
        echo_idx = torch.randint(low=0, high=N_echoes, size=(B,))
        echo_img = x_full[torch.arange(B), echo_idx, :, :].unsqueeze(1)  # (B,1,H,W)

        # ---- TE values corresponding to chosen echo ----
        te_vals = TEs_all[torch.tensor(echo_idx, device=device)].view(B,1)  # (B,1)

        # ---- brain masks from original echo 1 ----
        echo1_img = x_full[:,0:1,:,:]  # always choose echo 1 for mask
        masks = build_brain_mask_batch(echo1_img)  # (B,H,W)

        # ---- mask the input before feeding into model ----
        masked_echo = echo_img * masks.unsqueeze(1).float()  # (B,1,H,W)

        optimizer.zero_grad()

        # ---- forward pass through FiLM model ----
        pred_params = model(masked_echo, te_vals)  # (B,2,H,W)

        # ---- physics-based reconstruction ----
        y_hat = flash_forward_single_te(pred_params, te_vals.view(B,))  # (B,1,H,W)

        # ---- masked loss ----
        mask_exp = masks.unsqueeze(1).float()  # (B,1,H,W)
        loss = ((y_hat - echo_img)**2 * mask_exp).sum() / mask_exp.sum()

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * B

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # ----------------------------
    # VALIDATION
    # ----------------------------
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for x_full_val, _ in val_loader:
            x_full_val = x_full_val.to(device)
            Bv, N_echoes_v, Hv, Wv = x_full_val.shape

            echo_idx = torch.randint(low=0, high=N_echoes_v, size=(Bv,))
            echo_img = x_full_val[torch.arange(Bv), echo_idx, :, :].unsqueeze(1)
            te_vals = TEs_all[torch.tensor(echo_idx, device=device)].view(Bv,1)

            # always use echo 1 for mask
            echo1_val = x_full_val[:,0:1,:,:]
            masks = build_brain_mask_batch(echo1_val)

            masked_echo = echo_img * masks.unsqueeze(1).float()

            pred_params = model(masked_echo, te_vals)
            y_hat = flash_forward_single_te(pred_params, te_vals.view(Bv,))

            mask_exp = masks.unsqueeze(1).float()
            loss = ((y_hat - echo_img)**2 * mask_exp).sum() / mask_exp.sum()
            val_running += loss.item() * Bv

    val_loss = val_running / len(val_loader.dataset)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_network4_unet.pth"))
# ------------------------------------------------
# EVALUATION, SAVE, AND VISUALIZE (brain-masked, mask from echo 1)
# ------------------------------------------------
ECHO_INDICES = [1,2,3,4]
SAVE_EVAL_DIR = os.path.join(SAVE_DIR, "nifti_outputs")
os.makedirs(SAVE_EVAL_DIR, exist_ok=True)

model.eval()
with torch.no_grad():
    for subj in VAL_SUBJECTS[:10]:  # visualize a few subjects
        # Dataset for this subject
        ds = FlashMRIDataset([subj], DATA_DIR, echo_indices=ECHO_INDICES, mode="val")
        ls_params = load_ls_params(subj)  # (2,H,W,D)

        pred_t2s_slices = []
        pred_t1rho_slices = []
        pred_resid_slices = []

        mask_ref = None  # will store brain mask from echo 1

        for i in range(len(ds)):
            x_full, _ = ds[i]                 # (N_ECHOES,H,W)
            x_full = x_full.unsqueeze(0).to(device)  # (1,N_ECHOES,H,W)

            # ---- always pick echo 1 for brain mask ----
            echo_img_mask = x_full[:, 0:1, :, :]      # (1,1,H,W)
            if mask_ref is None:
                mask_ref = build_brain_mask_batch(echo_img_mask)[0].cpu().numpy()  # (H,W)

            # ---- pick random echo for prediction ----
            slice_residuals = []  # will hold (H,W) residuals for all echoes

            for e_idx in range(x_full.shape[1]):
                echo_img = x_full[:, e_idx:e_idx+1, :, :]   # (1,1,H,W)
                te_val = torch.tensor([[TEs_all[e_idx]]], device=device)

                pred_params = model(echo_img, te_val)       # (1,2,H,W)
                y_hat = flash_forward_single_te(pred_params, te_val.view(1))

                resid = (y_hat - echo_img).cpu().numpy().squeeze()  # (H,W)
                slice_residuals.append(resid)
                pred_t2s_slices.append(
                    pred_params[0, 0].cpu().numpy()
                )
                pred_t1rho_slices.append(
                    pred_params[0, 1].cpu().numpy()
                )

            # stack echoes → (H,W,4)
            slice_residuals = np.stack(slice_residuals, axis=-1)

            # apply mask → (H,W,4)
            slice_residuals = slice_residuals * mask_ref[..., np.newaxis]

            pred_resid_slices.append(slice_residuals)

            
            

        # Stack slices along z-axis
        pred_t2s_vol   = np.stack(pred_t2s_slices, axis=-1)
        pred_t1rho_vol = np.stack(pred_t1rho_slices, axis=-1)
        pred_resid_vol = np.stack(pred_resid_slices, axis=-1)
        pred_resid_vol = np.transpose(pred_resid_vol, (0, 1, 3, 2))

        # ------------------------
        # Save as NIfTI
        # ------------------------
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(pred_t2s_vol, affine), os.path.join(SAVE_EVAL_DIR, f"{subj}_T2star_pred.nii.gz"))
        nib.save(nib.Nifti1Image(pred_t1rho_vol, affine), os.path.join(SAVE_EVAL_DIR, f"{subj}_T1rho_pred.nii.gz"))
        nib.save(
            nib.Nifti1Image(pred_resid_vol, affine),
            os.path.join(SAVE_EVAL_DIR, f"{subj}_residuals.nii.gz")
        )
        print(f"Saved NIfTI predicted maps for {subj} → {SAVE_EVAL_DIR}")

        # ------------------------
        # Visualize middle slice (masked)
        # ------------------------
        mid_slice = len(pred_t2s_slices) // 2  # middle slice
        fig, axes = plt.subplots(2, 3, figsize=(12,6))

        def masked_show(ax, img, mask, cmap='hot'):
            img_masked = np.where(mask, img, np.nan)
            ax.imshow(img_masked, cmap=cmap, aspect='equal')
            ax.axis('off')

        # T2* predicted vs LS
        masked_show(axes[0,0], pred_t2s_slices[mid_slice], mask_ref)
        axes[0,0].set_title("T2* Pred (brain only)")
        masked_show(axes[0,1], ls_params[0][:,:,mid_slice], mask_ref)
        axes[0,1].set_title("T2* LS")
        masked_show(axes[0,2], np.abs(pred_t2s_slices[mid_slice]-ls_params[0][:,:,mid_slice]), mask_ref, cmap='gray')
        axes[0,2].set_title("T2* Residual")

        # T1ρ predicted vs LS
        masked_show(axes[1,0], pred_t1rho_slices[mid_slice], mask_ref)
        axes[1,0].set_title("T1ρ Pred (brain only)")
        masked_show(axes[1,1], ls_params[1][:,:,mid_slice], mask_ref)
        axes[1,1].set_title("T1ρ LS")
        masked_show(axes[1,2], np.abs(pred_t1rho_slices[mid_slice]-ls_params[1][:,:,mid_slice]), mask_ref, cmap='gray')
        axes[1,2].set_title("T1ρ Residual")

        plt.suptitle(f"{subj} | Middle Slice Comparison (brain mask applied from echo 1)")
        plt.tight_layout()

        # ------------------------------------------------
        # Save visualizations as PNG
        # ------------------------------------------------
        png_save_dir = os.path.join(SAVE_DIR, "png_outputs")
        os.makedirs(png_save_dir, exist_ok=True)

        # Save figure as PNG
        fig_path = os.path.join(png_save_dir, f"{subj}_middle_slice.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)  # close to free memory
        print(f"Saved middle slice visualization for {subj} → {fig_path}")
