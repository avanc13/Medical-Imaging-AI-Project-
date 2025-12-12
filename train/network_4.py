#!/usr/bin/env python3
"""
Network 4: Unsupervised with changing acquisition parameters and
TE as additional input.

- Each training sample: (single echo image, TE scalar)
- Network input: [echo_image, TE_map]  -> (B, 2, H, W)
- Network output: [T2*, T1rho] param maps
- Physics model: y_hat(TE) = T1rho * exp(-TE / T2*)
- Loss: masked MSE between y_hat(TE_input) and that echo, using brain mask
        (same masking strategy as Network 1).

We randomize which echo (TE) is used per slice and per epoch, so the
network sees multiple contrasts and learns TE-aware parameter estimation.
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
from dataloaders.flash_dataset import FlashMRIDataset
from models.avantika.unet import UNet

# ------------------------------------------------
# DEVICE
# ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------
# DATA CONFIG
# ------------------------------------------------
DATA_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data/processed"

# discover subjects
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

np.save("train_subjects.npy", np.array(TRAIN_SUBJECTS))
np.save("val_subjects.npy", np.array(VAL_SUBJECTS))

# All echoes present
ECHO_INDICES_ALL = [1, 2, 3, 4]
N_ECHOES = len(ECHO_INDICES_ALL)

N_PARAMS   = 2   # [T2*, T1rho]
BATCH_SIZE = 4
NUM_EPOCHS = 100
LR         = 1e-4
SAVE_DIR   = "redoing_stuff_12_11/checkpoints_network4_with_brainmask"
os.makedirs(SAVE_DIR, exist_ok=True)
MASK_PERCENTILE = 60.0  # same as Net1/2

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

def build_brain_mask_2d_batch(x, percentile: float = MASK_PERCENTILE):
    """
    Build a per-slice brain mask from the first echo in the batch.

    x: (B, N_echoes, H, W), already normalized echoes
    Returns: mask of shape (B, N_echoes, H, W) with 1 inside brain, 0 outside.
    """
    # we use only echo 1 to define the brain for each slice
    B, C, H, W = x.shape
    echo1 = x[:, 0, :, :]  # (B, H, W)

    masks = []
    for b in range(B):
        e = echo1[b].detach().cpu().numpy()  # (H, W)

        nonzero_vals = e[e > 0]
        if nonzero_vals.size == 0:
            # degenerate slice, just give zeros mask
            masks.append(np.zeros_like(e, dtype=np.float32))
            continue

        thr = np.percentile(nonzero_vals, percentile)
        m = e > thr

        # 2D morphology
        m = binary_closing(m, structure=np.ones((3, 3)))
        m = binary_fill_holes(m)

        # largest connected component
        lbl = label(m)
        if lbl.max() > 0:
            largest_cc = np.argmax(np.bincount(lbl.ravel()[1:])) + 1
            m = (lbl == largest_cc)

        masks.append(m.astype(np.float32))

    mask_np = np.stack(masks, axis=0)  # (B, H, W)
    mask = torch.from_numpy(mask_np).to(x.device)  # (B, H, W)
    mask = mask.unsqueeze(1)  # (B, 1, H, W)
    mask = mask.expand(-1, C, -1, -1)  # (B, N_echoes, H, W)

    return mask


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

# ------------------------------------------------
# MODEL / LOSS / OPTIMIZER
# ------------------------------------------------
# Input channels: 1 (echo image) + 1 (TE map)
model = UNet(in_channels=2, out_channels=N_PARAMS).to(device)
#criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                 factor=0.5, patience=5)

train_losses = []
val_losses   = []
best_val_loss = float("inf")

# ------------------------------------------------
# SHAPE SANITY CHECK
# ------------------------------------------------
print("\n=== Network 4 shape check ===")
x_full, sid = next(iter(train_loader))  # x_full: (B, N_echoes, H, W)
x_full = x_full.to(device)
print(f"Batch subjects: {sid[:4]}")
print(f"x_full shape (all echoes): {x_full.shape}")

B, C, H, W = x_full.shape

# pretend we randomly pick one echo per sample
rand_idx = torch.randint(low=0, high=C, size=(B,))
x_echo   = x_full[torch.arange(B), rand_idx, :, :].unsqueeze(1)  # (B,1,H,W)
te_vals  = TEs_all[rand_idx].to(device)                          # (B,)


te_map = te_vals.view(B, 1, 1, 1).expand(-1, 1, H, W)            # (B,1,H,W) 
x_in   = torch.cat([x_echo, te_map], dim=1)                    # (B,2,H,W)

print(f"x_echo shape: {x_echo.shape}")
print(f"x_in   shape (echo+TE): {x_in.shape}")

#x_in = x_in.to(device)
params_pred = model(x_in)                                        # (B,2,H,W)
print(f"params_pred shape: {params_pred.shape}")

y_hat = flash_forward_single_te(params_pred, te_vals.to(device)) # (B,1,H,W)
print(f"y_hat shape: {y_hat.shape}")
print("=== shape check done ===\n")

# ------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for batch_idx, (x_full, _) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
    ):
        x_full = x_full.to(device)  # (B, N_echoes, H, W)
        B, C, H, W = x_full.shape

        # Randomly choose one echo per sample
        rand_idx = torch.randint(low=0, high=C, size=(B,), device=device)  # 0..C-1
        x_echo   = x_full[torch.arange(B, device=device), rand_idx, :, :].unsqueeze(1)  # (B,1,H,W)
        te_vals  = TEs_all[rand_idx]  # (B,)

        # Build TE map channel
        te_map = te_vals.view(B, 1, 1, 1).expand(-1, 1, H, W)  # (B,1,H,W)

        # Network input: echo image + TE map
        x_in = torch.cat([x_echo, te_map], dim=1)  # (B,2,H,W)

        params_pred = model(x_in)                               # (B,2,H,W)
        y_hat       = flash_forward_single_te(params_pred, te_vals)  # (B,1,H,W)

        if epoch == 0 and batch_idx == 0:
            print("Input echo range:", x_echo.min().item(), x_echo.max().item())
            print("TE range (s):    ", te_vals.min().item(), te_vals.max().item())
            print("Pred T2* range:  ", params_pred[:,0].min().item(), params_pred[:,0].max().item())
            print("Pred T1rho range:", params_pred[:,1].min().item(), params_pred[:,1].max().item())
            print("y_hat range:     ", y_hat.min().item(), y_hat.max().item())

        #loss = criterion(y_hat, x_echo)  # compare to that single echo
        # -------- masked brain-only loss (same idea as Net 1) --------
        # build brain mask from first echo in x_full
        mask_full = build_brain_mask_2d_batch(x_full)  # (B, N_echoes, H, W)
        # Take mask corresponding to the chosen echo_idx per sample
        mask_echo = mask_full[torch.arange(B, device=device), rand_idx, :, :].unsqueeze(1)  # (B,1,H,W)

        diff2 = (y_hat - x_echo) ** 2
        diff2 = diff2 * mask_echo
        loss = diff2.sum() / (mask_echo.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # ----------------------------
    # VALIDATION
    # ----------------------------
    model.eval()
    val_loss = 0.0
    val_nmse = 0.0
    with torch.no_grad():
        for x_full_val, _ in val_loader:
            x_full_val = x_full_val.to(device)  # (B,N_echoes,H,W)
            Bv, Cv, Hv, Wv = x_full_val.shape

            # For validation, evaluate loss averaged over *all* echoes
            mask_full_val = build_brain_mask_2d_batch(x_full_val)  # (Bv,Cv,Hv,Wv)

            batch_loss = 0.0
            batch_nmse = 0.0

            for echo_idx in range(Cv):
                x_echo_v  = x_full_val[:, echo_idx, :, :].unsqueeze(1)    # (Bv,1,Hv,Wv)
                te_vals_v = TEs_all[echo_idx].expand(Bv)                  # (Bv,)

                te_map_v  = te_vals_v.view(Bv,1,1,1).expand(-1,1,Hv,Wv)   # (Bv,1,Hv,Wv)
                x_in_v    = torch.cat([x_echo_v, te_map_v], dim=1)        # (Bv,2,Hv,Wv)

                params_v  = model(x_in_v)
                y_hat_v   = flash_forward_single_te(params_v, te_vals_v)  # (Bv,1,Hv,Wv)

                # mask for this echo
                mask_echo_v = mask_full_val[:, echo_idx, :, :].unsqueeze(1)  # (Bv,1,Hv,Wv)

                diff2_v = (y_hat_v - x_echo_v) ** 2
                diff2_v = diff2_v * mask_echo_v
                batch_loss += (diff2_v.sum() / (mask_echo_v.sum() + 1e-8)).item()

                # NMSE also inside brain only
                batch_nmse += nmse(y_hat_v * mask_echo_v, x_echo_v * mask_echo_v).item()


            batch_loss /= Cv
            batch_nmse /= Cv

            val_loss += batch_loss
            val_nmse += batch_nmse

    val_loss /= len(val_loader)
    val_nmse /= len(val_loader)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {train_loss:.6f} | "
          f"Val Loss: {val_loss:.6f} | "
          f"Val NMSE: {val_nmse:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, "best_network4_unet.pth"))

# ------------------------------------------------
# export NIFTI for all val subjects
# ------------------------------------------------
SAVE_NIFTI_DIR = os.path.join(SAVE_DIR, "nifti_outputs")
os.makedirs(SAVE_NIFTI_DIR, exist_ok=True)

model.eval()
with torch.no_grad():
    for subj in VAL_SUBJECTS:
        ds = FlashMRIDataset([subj], DATA_DIR,
                             echo_indices=ECHO_INDICES_ALL,
                             mode="val")

        all_t2s, all_t1rho = [], []
        all_recons, all_inputs = [], []

        for i in range(len(ds)):
            x_full_slice, _ = ds[i]           # (N_echoes, H, W)
            x_full_slice = x_full_slice.unsqueeze(0).to(device)  # (1,N_echoes,H,W)
            _, Ce, He, We = x_full_slice.shape

            # --- estimate params for each echo and average ---
            params_per_echo = []
            for echo_idx in range(Ce):
                x_echo_s  = x_full_slice[:, echo_idx, :, :].unsqueeze(1) # (1,1,H,W)
                te_val_s  = TEs_all[echo_idx].view(1)                    # (1,)

                te_map_s  = te_val_s.view(1,1,1,1).expand(-1,1,He,We)    # (1,1,H,W)
                x_in_s    = torch.cat([x_echo_s, te_map_s], dim=1)       # (1,2,H,W)

                params_s  = model(x_in_s)                                # (1,2,H,W)
                params_per_echo.append(params_s)

            # average params over echoes (TE-invariant tissue parameters)
            params_stack = torch.stack(params_per_echo, dim=0)           # (Ce,1,2,H,W)
            params_mean  = params_stack.mean(dim=0).squeeze(0)           # (2,H,W)

            # reconstruct all 4 echoes from averaged params
            params_mean = params_mean.unsqueeze(0)                        # (1,2,H,W)
            T2s_map   = torch.abs(params_mean[:,0]).cpu().numpy().squeeze()
            T1rho_map = torch.abs(params_mean[:,1]).cpu().numpy().squeeze()

            # reconstruct echoes as in Network 1/3
            recons = []
            inputs = []
            for echo_idx in range(Ce):
                te_val_s  = TEs_all[echo_idx].view(1)                    # (1,)
                y_hat_s   = flash_forward_single_te(params_mean, te_val_s)  # (1,1,H,W)

                recon_e   = y_hat_s.cpu().numpy().squeeze()              # (H,W)
                inp_e     = x_full_slice[:, echo_idx, :, :].cpu().numpy().squeeze()

                recons.append(recon_e)
                inputs.append(inp_e)

            all_t2s.append(T2s_map)
            all_t1rho.append(T1rho_map)
            all_recons.append(np.stack(recons, axis=0))   # (N_echoes,H,W)
            all_inputs.append(np.stack(inputs, axis=0))   # (N_echoes,H,W)

        # stack over slices in z
        t2s_vol   = np.stack(all_t2s, axis=-1)          # (H,W,Z)
        t1rho_vol = np.stack(all_t1rho, axis=-1)        # (H,W,Z)

        recon_vol = np.stack(all_recons, axis=-1)       # (N_echoes,H,W,Z)
        input_vol = np.stack(all_inputs, axis=-1)       # (N_echoes,H,W,Z)
        resid_vol = np.abs(input_vol - recon_vol)       # (N_echoes,H,W,Z)

        # move echoes last
        recon_vol = np.transpose(recon_vol, (1,2,3,0))  # (H,W,Z,N_echoes)
        input_vol = np.transpose(input_vol, (1,2,3,0))
        resid_vol = np.transpose(resid_vol, (1,2,3,0))

        affine = np.eye(4)
        nib.save(nib.Nifti1Image(t2s_vol,   affine),
                 os.path.join(SAVE_NIFTI_DIR, f"{subj}_T2star_pred.nii.gz"))
        nib.save(nib.Nifti1Image(t1rho_vol, affine),
                 os.path.join(SAVE_NIFTI_DIR, f"{subj}_T1rho_pred.nii.gz"))
        nib.save(nib.Nifti1Image(recon_vol, affine),
                 os.path.join(SAVE_NIFTI_DIR, f"{subj}_reconstructed_echoes.nii.gz"))
        nib.save(nib.Nifti1Image(resid_vol, affine),
                 os.path.join(SAVE_NIFTI_DIR, f"{subj}_residuals.nii.gz"))

        print(f"Saved NIfTI outputs for {subj} → {SAVE_NIFTI_DIR}")

# ------------------------------------------------
# PLOT LOSSES
# ------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_losses, label="Validation Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Network 4: TE-conditioned self-supervised training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "loss_curve_network4.png"), dpi=150)
plt.close()

print(f"Training complete. Best val loss: {best_val_loss:.6f}")
