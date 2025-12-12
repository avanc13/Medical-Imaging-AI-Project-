#!/usr/bin/env python3
"""
Supervised training script (Network 2) for multi-echo FLASH MRI.

Trains a 2D U-Net to predict per-pixel T2* and T1rho maps from
multi-echo input, using LS maps as targets. Also passes the
predicted params through the FLASH forward model so we can compare
reconstruction error with Network 1/3/4.

-- this is gonna inlcude brain maskiing too

To change:
    - DATA_DIR: clean vs noisy processed echoes
    - ECHO_INDICES: which echoes to use as input
    - SAVE_DIR: where to put checkpoints + NIfTI outputs
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.measure import label

from models.avantika.unet import UNet

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# location of processed echo .npy inputs
DATA_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data/processed_noisy"

# location of the ground-truth LS maps
PARAM_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/ground_truth_maps"

# Normalization constants for ground-truth parameter maps  # <<< NEW >>>
T2S_MEAN   = 0.1175
T2S_STD    = 0.1665
T1P_MEAN   = 358.4492
T1P_STD    = 626.1934

# choose which echoes to use as input (1-based indices)
ECHO_INDICES = [1, 2, 3, 4]
N_ECHOES = len(ECHO_INDICES)

N_PARAMS = 2   # [T2*, T1rho]
BATCH_SIZE = 4
NUM_EPOCHS = 100
LR = 1e-4

MASK_PERCENTILE = 60.0  

SAVE_DIR = "redoing_stuff_12_11/checkpoints_network2_allechoes_with_brainmask_noisy"
os.makedirs(SAVE_DIR, exist_ok=True)

# echo times (seconds) from metadata
TEs_all = torch.tensor([0.012, 0.028, 0.044, 0.060], device=device)
TEs = TEs_all[[i - 1 for i in ECHO_INDICES]]


# ------------------------------------------------
# FLASH PHYSICS MODEL
# ------------------------------------------------
def flash_forward(params: torch.Tensor, TEs: torch.Tensor) -> torch.Tensor:
    """
    y(TE) = (T1rho) * exp(-TE / T2*)
    params: (B, 2, H, W) → [T2*, T1rho] in PHYSICAL UNITS
    """
    T2s   = torch.abs(params[:, 0:1, :, :]) + 1e-3
    T1rho = torch.abs(params[:, 1:2, :, :]) + 1e-3
    y_hat = T1rho * torch.exp(-TEs.view(1, -1, 1, 1) / T2s)
    return y_hat

# same brain mask as network 1
def build_brain_mask_2d_batch(x: torch.Tensor,
                              percentile: float = MASK_PERCENTILE) -> torch.Tensor:
    """
    Build a per-slice brain mask from the first echo in the batch.

    x: (B, C, H, W), already normalized echoes
    Returns: mask of shape (B, 1, H, W) with 1 inside brain, 0 outside.
    """
    B, C, H, W = x.shape
    echo1 = x[:, 0, :, :]  # (B, H, W)

    masks = []
    for b in range(B):
        e = echo1[b].detach().cpu().numpy()

        nonzero_vals = e[e > 0]
        if nonzero_vals.size == 0:
            masks.append(np.zeros_like(e, dtype=np.float32))
            continue

        thr = np.percentile(nonzero_vals, percentile)
        m = e > thr

        m = binary_closing(m, structure=np.ones((3, 3)))
        m = binary_fill_holes(m)

        lbl = label(m)
        if lbl.max() > 0:
            largest_cc = np.argmax(np.bincount(lbl.ravel()[1:])) + 1
            m = (lbl == largest_cc)

        masks.append(m.astype(np.float32))

    mask_np = np.stack(masks, axis=0)  # (B, H, W)
    mask = torch.from_numpy(mask_np).to(x.device)  # (B, H, W)
    mask = mask.unsqueeze(1)  # (B, 1, H, W)
    return mask  # (B, 1, H, W)

# ------------------------------------------------
# LS PARAMETER MAP LOADER
# ------------------------------------------------
def load_ls_params(subj_id):
    """
    Load least-squares parameter maps for one subject.

    Assumes:
        T2*:   sub-XXXXX_T2star.npy   -> (H, W, D)
        T1rho: sub-XXXXX_T1p.npy      -> (H, W, D)

    Returns:
        params_np: np.ndarray of shape (2, H, W, D)
                   index 0 -> T2*
                   index 1 -> T1rho
    """
    t2s_path = os.path.join(PARAM_DIR, f"{subj_id}_T2star.npy")
    t1r_path = os.path.join(PARAM_DIR, f"{subj_id}_T1p.npy")

    if not os.path.exists(t2s_path):
        raise FileNotFoundError(f"Missing LS T2* map: {t2s_path}")
    if not os.path.exists(t1r_path):
        raise FileNotFoundError(f"Missing LS T1rho map: {t1r_path}")

    t2s   = np.load(t2s_path)   # (H, W, D), physical units
    t1rho = np.load(t1r_path)   # (H, W, D), physical units

    params = np.stack([t2s, t1rho], axis=0)  # (2, H, W, D)
    return params


# ------------------------------------------------
# SUPERVISED DATASET
# ------------------------------------------------
class FlashMRIDatasetSupervised(torch.utils.data.Dataset):
    """
    Supervised wrapper for Network 2.

    Returns (per slice):
        x    : (C, H, W)   multi-echo input
        p_gt : (2, H, W)   [T2*, T1rho] **normalized**
    """
    def __init__(self, subject_ids, data_dir, param_dir, echo_indices=None):
        self.subject_ids = subject_ids
        self.data_dir = data_dir
        self.param_dir = param_dir
        self.echo_indices = echo_indices or [1, 2, 3, 4]

        # build (subject, slice_idx) index
        self.index = []
        for subj_id in subject_ids:
            first_echo_path = os.path.join(
                data_dir, f"{subj_id}_echo{self.echo_indices[0]}.npy"
            )
            if not os.path.exists(first_echo_path):
                raise FileNotFoundError(
                    f"Missing echo file for {subj_id}: {first_echo_path}"
                )
            depth = np.load(first_echo_path, mmap_mode="r").shape[2]
            for z in range(depth):
                self.index.append((subj_id, z))

        print(
            f"[SUPERVISED] dataset built with {len(self.index)} slices "
            f"from {len(subject_ids)} subjects."
        )

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
            vol = np.load(path)  # (H, W, D), typically already intensity-normalized
            vols.append(vol)
        stacked = np.stack(vols, axis=-1)  # (H, W, D, N_echoes)
        return stacked

    def _load_subject_params(self, subj_id):
        if subj_id in self._param_cache:
            return self._param_cache[subj_id]
        params = load_ls_params(subj_id)  # (2, H, W, D), physical units
        self._param_cache[subj_id] = params
        return params

    def __getitem__(self, idx):
        subj_id, z = self.index[idx]

        # multi-echo inputs
        if subj_id in self._volume_cache:
            vol = self._volume_cache[subj_id]
        else:
            vol = self._load_subject_inputs(subj_id)
            self._volume_cache[subj_id] = vol

        # LS parameter maps (physical units)
        params = self._load_subject_params(subj_id)  # (2, H, W, D)

        # slice along depth
        slice_img   = vol[:, :, z, :]      # (H, W, N_echoes)
        slice_param = params[:, :, :, z]   # (2, H, W) physical

        slice_param_phys = slice_param  # (2, H, W)

        x    = torch.from_numpy(slice_img).permute(2, 0, 1).float()
        p_gt = torch.from_numpy(slice_param_phys).float()


        return x, p_gt


def nmse(pred, target, eps: float = 1e-8) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

    pred = pred.float()
    target = target.float()

    batch_dims = pred.shape[0]
    pred_flat = pred.view(batch_dims, -1)
    target_flat = target.view(batch_dims, -1)

    diff = pred_flat - target_flat
    num = (diff * diff).sum(dim=1)
    denom = (target_flat * target_flat).sum(dim=1)

    nmse_per_sample = num / (denom + eps)
    return nmse_per_sample.mean()


# ------------------------------------------------
# SUBJECT SPLIT
# ------------------------------------------------
subjects = sorted(
    set(
        os.path.basename(f).split("_echo")[0]
        for f in glob.glob(os.path.join(DATA_DIR, "sub-*_echo1.npy"))
    )
)
print(f"Found {len(subjects)} subjects total.")

# remove corrupted
subjects = [s for s in subjects if s != "sub-04620"]
print(f"Remaining {len(subjects)} subjects after removing corrupted entries.")

train_subj, val_subj = train_test_split(
    subjects, test_size=0.125, random_state=42
)
print(
    f"Training on {len(train_subj)} subjects, "
    f"validating on {len(val_subj)} subjects."
)

TRAIN_SUBJECTS = train_subj
VAL_SUBJECTS = val_subj

np.save("train_subjects.npy", np.array(TRAIN_SUBJECTS))
np.save("val_subjects.npy", np.array(VAL_SUBJECTS))



# ------------------------------------------------
# DATA LOADERS
# ------------------------------------------------
train_ds = FlashMRIDatasetSupervised(
    TRAIN_SUBJECTS, DATA_DIR, PARAM_DIR, echo_indices=ECHO_INDICES
)
val_ds = FlashMRIDatasetSupervised(
    VAL_SUBJECTS, DATA_DIR, PARAM_DIR, echo_indices=ECHO_INDICES
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4)


# ------------------------------------------------
# MODEL / LOSS / OPTIMIZER
# ------------------------------------------------
model = UNet(in_channels=N_ECHOES, out_channels=N_PARAMS).to(device)
#criterion = nn.MSELoss()  # supervised loss on *normalized* parameter maps
# new mased loss that only computes loss inside brain mask
def masked_mse_params(p_pred, p_gt, mask_params):
    """
    p_pred, p_gt: (B, 2, H, W)
    mask_params: (B, 2, H, W) with 1 inside brain, 0 outside
    """
    diff2 = (p_pred - p_gt) ** 2
    diff2 = diff2 * mask_params
    return diff2.sum() / (mask_params.sum() + 1e-8)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

train_losses, val_losses = [], []
best_val_loss = float("inf")

# quick shape sanity-check
print("\n=== verifying shapes for Network 2 ===")
x_test, p_test = next(iter(train_loader))
print("Input batch shape :", x_test.shape)   # [B, N_ECHOES, H, W]
print("GT params shape   :", p_test.shape)   # [B, 2, H, W] (normalized)
with torch.no_grad():
    y_test = model(x_test.to(device))
print("Predicted shape   :", y_test.shape)   # [B, 2, H, W] (normalized)
print("make sure above are consistent.\n")


# ------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for batch_idx, (x, p_gt) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
    ):
        x    = x.to(device)
        p_gt = p_gt.to(device)  # normalized GT

        brain_mask = build_brain_mask_2d_batch(x)              # (B, 1, H, W)
        mask_params = brain_mask.expand(-1, N_PARAMS, -1, -1)  # (B, 2, H, W)

        p_pred = model(x)
        loss   = masked_mse_params(p_pred, p_gt, mask_params)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if epoch == 0 and batch_idx == 0:
            print("Batch 0 stats (normalized):")
            print("  p_gt T2* range      :", p_gt[:, 0].min().item(), p_gt[:, 0].max().item())
            print("  p_gt T1rho range    :", p_gt[:, 1].min().item(), p_gt[:, 1].max().item())
            print("  p_pred T2* range    :", p_pred[:, 0].min().item(), p_pred[:, 0].max().item())
            print("  p_pred T1rho range  :", p_pred[:, 1].min().item(), p_pred[:, 1].max().item())

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # ------------------------------------------------
    # VALIDATION
    # ------------------------------------------------
    model.eval()
    val_loss = 0.0
    val_nmse_params = 0.0   # NMSE in normalized space
    val_nmse_recon  = 0.0   # NMSE on echoes via physics model

    with torch.no_grad():
        for x_val, p_gt_val in val_loader:
            x_val    = x_val.to(device)
            p_gt_val = p_gt_val.to(device)

            brain_mask_val = build_brain_mask_2d_batch(x_val)           # (B, 1, H, W)
            mask_params_val = brain_mask_val.expand(-1, N_PARAMS, -1, -1)  # (B, 2, H, W)
            mask_echo_val   = brain_mask_val.expand(-1, N_ECHOES, -1, -1)  # (B, C, H, W)

            p_pred_val = model(x_val)
            loss = masked_mse_params(p_pred_val, p_gt_val, mask_params_val)
            val_loss += loss.item()

            # NMSE on params inside brain
            val_nmse_params += nmse(p_pred_val * mask_params_val,
                                    p_gt_val * mask_params_val).item()

            # recon via physics model, NMSE inside brain
            p_pred_phys = p_pred_val  # already physical units
            y_hat_val   = flash_forward(p_pred_phys, TEs)
            val_nmse_recon += nmse(y_hat_val * mask_echo_val,
                                   x_val * mask_echo_val).item()

    val_loss /= len(val_loader)
    val_nmse_params /= len(val_loader)
    val_nmse_recon  /= len(val_loader)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Param Loss (norm): {val_loss:.6f} | "
        f"Val Param NMSE (norm): {val_nmse_params:.6f} | "
        f"Val Recon NMSE: {val_nmse_recon:.6f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, "best_network2_unet.pth"),
        )


# ------------------------------------------------
# SAVE SOME NIFTI OUTPUTS (PRED PARAMS + RECONS + RESIDUALS)
# ------------------------------------------------
SAVE_NIFTI_DIR = os.path.join(SAVE_DIR, "nifti_outputs")
os.makedirs(SAVE_NIFTI_DIR, exist_ok=True)

model.eval()
with torch.no_grad():
    for subj in VAL_SUBJECTS:
        ds_subj = FlashMRIDatasetSupervised(
            [subj], DATA_DIR, PARAM_DIR, echo_indices=ECHO_INDICES
        )

        all_t2s_pred = []
        all_t1rho_pred = []
        all_recons = []
        all_inputs = []

        for i in range(len(ds_subj)):
            x_slice, _ = ds_subj[i]
            x_slice_b = x_slice.unsqueeze(0).to(device)

            p_pred_phys = model(x_slice_b)                     # (1, 2, H, W)
            y_hat  = flash_forward(p_pred_phys, TEs)           # (1, N_ECHOES, H, W)

            t2s_pred   = torch.abs(p_pred_phys[:, 0]).cpu().numpy().squeeze()
            t1rho_pred = torch.abs(p_pred_phys[:, 1]).cpu().numpy().squeeze()
            recon      = y_hat.cpu().numpy().squeeze()         # (N_ECHOES, H, W)
            inp        = x_slice_b.cpu().numpy().squeeze()     # (N_ECHOES, H, W)

            all_t2s_pred.append(t2s_pred)
            all_t1rho_pred.append(t1rho_pred)
            all_recons.append(recon)
            all_inputs.append(inp)

        t2s_vol   = np.stack(all_t2s_pred, axis=-1)
        t1rho_vol = np.stack(all_t1rho_pred, axis=-1)

        recon_vol = np.stack(all_recons, axis=-1)   # (N_ECHOES, H, W, Z)
        input_vol = np.stack(all_inputs, axis=-1)
        resid_vol = np.abs(input_vol - recon_vol)

        recon_vol = np.transpose(recon_vol, (1, 2, 3, 0))
        input_vol = np.transpose(input_vol, (1, 2, 3, 0))
        resid_vol = np.transpose(resid_vol, (1, 2, 3, 0))

        affine = np.eye(4)
        nib.save(
            nib.Nifti1Image(t2s_vol, affine),
            os.path.join(SAVE_NIFTI_DIR, f"{subj}_T2star_pred.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(t1rho_vol, affine),
            os.path.join(SAVE_NIFTI_DIR, f"{subj}_T1rho_pred.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(recon_vol, affine),
            os.path.join(SAVE_NIFTI_DIR, f"{subj}_reconstructed_echoes.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(resid_vol, affine),
            os.path.join(SAVE_NIFTI_DIR, f"{subj}_residuals.nii.gz"),
        )

        print(f"Saved NIfTI outputs for {subj} → {SAVE_NIFTI_DIR}")


# ------------------------------------------------
# LOSS CURVE
# ------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss (masked)", linewidth=2)
plt.plot(val_losses,   label="Validation Loss (masked)", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (masked params)")
plt.title("Supervised Training (Network 2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "loss_curve_network2.png"), dpi=150)
plt.show()

print(f"Training complete. Best val loss (masked params): {best_val_loss:.6f}")







