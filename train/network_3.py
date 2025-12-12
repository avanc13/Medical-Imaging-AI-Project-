#!/usr/bin/env python3
"""
Network 3 training script (unsupervised / TE-mismatched).

Idea:
- Input: a subset of echoes at some TEs_in.
- Model: 2D U-Net predicting parameter maps [T2*, T1rho].
- Physics: FLASH forward model y(TE) = T1rho * exp(-TE / T2*).
- Loss: MSE between reconstructed echoes at (possibly different) TEs_tgt
        and the measured echoes at those TEs.

This lets you train the model to extrapolate / interpolate in TE:
you can feed some echoes and train it to predict others.
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

# all subjects that have echo1
subjects = sorted(set([
    os.path.basename(f).split("_echo")[0]
    for f in glob.glob(os.path.join(DATA_DIR, "sub-*_echo1.npy"))
]))
print(f"Found {len(subjects)} subjects total.")

# drop known-bad subject(s)
subjects = [s for s in subjects if s != "sub-04620"]
print(f"Remaining {len(subjects)} subjects after removing corrupted entries.")

train_subj, val_subj = train_test_split(subjects, test_size=0.125, random_state=42)
print(f"Training on {len(train_subj)} subjects, validating on {len(val_subj)} subjects.")

TRAIN_SUBJECTS = train_subj
VAL_SUBJECTS   = val_subj

np.save("train_subjects.npy", np.array(TRAIN_SUBJECTS))
np.save("val_subjects.npy", np.array(VAL_SUBJECTS))

# ---- Echo configuration ----
# All echoes physically present in the data
ECHO_INDICES_ALL = [1, 2, 3, 4]

# Echoes that will be fed into the U-Net
# (change this to do experiments, e.g. [1,2], [1,3], etc.)
INPUT_ECHO_INDICES  = [1, 2]

# Echoes you want the model to reconstruct and match
# Often you'll use all, but you can also use a held-out set.
TARGET_ECHO_INDICES = [1, 2, 3, 4]

assert set(INPUT_ECHO_INDICES).issubset(ECHO_INDICES_ALL), "INPUT_ECHO_INDICES must be subset of ECHO_INDICES_ALL"
assert set(TARGET_ECHO_INDICES).issubset(ECHO_INDICES_ALL), "TARGET_ECHO_INDICES must be subset of ECHO_INDICES_ALL"

N_INPUT_ECHO   = len(INPUT_ECHO_INDICES)
N_TARGET_ECHO  = len(TARGET_ECHO_INDICES)
N_PARAMS       = 2   # [T2*, T1rho]

BATCH_SIZE     = 4
NUM_EPOCHS     = 100
LR             = 1e-4
SAVE_DIR       = "redoing_stuff_12_11/checkpoints_network3_with_brainmask"
os.makedirs(SAVE_DIR, exist_ok=True)

MASK_PERCENTILE = 60.0  # same as Net1/2

# Echo times (seconds) for all 4 echoes.
TEs_all = torch.tensor([0.012, 0.028, 0.044, 0.060], device=device)

# Map echo indices (1-based in your naming) to positions in TEs_all
def idx_positions(all_indices, subset):
    pos = []
    for e in subset:
        if e not in all_indices:
            raise ValueError(f"Echo {e} not in ECHO_INDICES_ALL={all_indices}")
        pos.append(all_indices.index(e))
    return pos

input_chan_pos  = idx_positions(ECHO_INDICES_ALL, INPUT_ECHO_INDICES)   # e.g. [0,1]
target_chan_pos = idx_positions(ECHO_INDICES_ALL, TARGET_ECHO_INDICES)  # e.g. [0,1,2,3]

TEs_input  = TEs_all[input_chan_pos]   # TEs corresponding to echoes fed into U-Net
TEs_target = TEs_all[target_chan_pos]  # TEs we reconstruct and compare against

print(f"INPUT_ECHO_INDICES:  {INPUT_ECHO_INDICES}, positions {input_chan_pos}, TEs_in  = {TEs_input.cpu().numpy()}")
print(f"TARGET_ECHO_INDICES: {TARGET_ECHO_INDICES}, positions {target_chan_pos}, TEs_tgt = {TEs_target.cpu().numpy()}")

# ------------------------------------------------
# PHYSICS MODEL
# ------------------------------------------------
def flash_forward(params, TEs):
    """
    FLASH MRI forward model:
        y(TE) = T1rho * exp(-TE / T2*)

    params: (B, 2, H, W) -> [T2*, T1rho]
    TEs:    (N_echoes,)
    Returns: (B, N_echoes, H, W)
    """
    T2s   = torch.abs(params[:, 0:1, :, :]) + 1e-3   # avoid zero / negative
    T1rho = torch.abs(params[:, 1:2, :, :]) + 1e-3
    y_hat = T1rho * torch.exp(-TEs.view(1, -1, 1, 1) / T2s)
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

    x: (B, N_all, H, W), normalized echoes
    Returns: (B, 1, H, W) mask with 1 inside brain, 0 outside.
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
    return mask


def masked_mse_echo(y_hat, x_tgt, mask_echo):
    """
    y_hat, x_tgt: (B, N_target, H, W)
    mask_echo:   (B, N_target, H, W) with 1 inside brain, 0 outside.
    """
    diff2 = (y_hat - x_tgt) ** 2
    diff2 = diff2 * mask_echo
    return diff2.sum() / (mask_echo.sum() + 1e-8)


# ------------------------------------------------
# DATASET / DATALOADERS
# ------------------------------------------------
# Dataset always loads ALL echoes listed in ECHO_INDICES_ALL; we subselect inside the loop.
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
model = UNet(in_channels=N_INPUT_ECHO, out_channels=N_PARAMS).to(device)
#criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                 factor=0.5, patience=5)

train_losses = []
val_losses   = []
best_val_loss = float("inf")

# Quick shape sanity check
print("\n=== verifying shapes (Network 3) ===")
x_full, sid = next(iter(train_loader))  # x_full: (B, N_all, H, W)
print(f"batch subjects: {sid[:4]}")
print(f"full input shape (all echoes): {x_full.shape}")

x_full = x_full.to(device)
x_in   = x_full[:, input_chan_pos,  :, :]   # (B, N_input, H, W)
x_tgt  = x_full[:, target_chan_pos, :, :]   # (B, N_target, H, W)
print(f"x_in shape  (to U-Net):      {x_in.shape}")
print(f"x_tgt shape (recon target): {x_tgt.shape}")

params_pred = model(x_in)                  # (B, 2, H, W)
print(f"Predicted params shape: {params_pred.shape}")

y_hat = flash_forward(params_pred, TEs_target)  # (B, N_target, H, W)
print(f"Reconstructed echoes shape (target TEs): {y_hat.shape}")
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
        x_full = x_full.to(device)                  # (B, N_all, H, W)
        x_in   = x_full[:, input_chan_pos,  :, :]   # (B, N_input, H, W)
        x_tgt  = x_full[:, target_chan_pos, :, :]   # (B, N_target, H, W)

        params_pred = model(x_in)
        y_hat       = flash_forward(params_pred, TEs_target)

        if epoch == 0 and batch_idx == 0:
            print("Input echoes (in) range:", x_in.min().item(), x_in.max().item())
            print("Target echoes range:    ", x_tgt.min().item(), x_tgt.max().item())
            print("Pred T2* range:         ", params_pred[:, 0].min().item(), params_pred[:, 0].max().item())
            print("Pred T1rho range:       ", params_pred[:, 1].min().item(), params_pred[:, 1].max().item())
            print("y_hat range:            ", y_hat.min().item(), y_hat.max().item())

        # --- brain mask from full echoes, expanded to target channels ---
        brain_mask   = build_brain_mask_2d_batch(x_full)                  # (B, 1, H, W)
        mask_echo    = brain_mask.expand(-1, N_TARGET_ECHO, -1, -1)       # (B, N_target, H, W)

        loss = masked_mse_echo(y_hat, x_tgt, mask_echo)


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
            x_full_val = x_full_val.to(device)
            x_in_val   = x_full_val[:, input_chan_pos,  :, :]
            x_tgt_val  = x_full_val[:, target_chan_pos, :, :]

            params_val = model(x_in_val)
            y_val_hat  = flash_forward(params_val, TEs_target)

            brain_mask_val = build_brain_mask_2d_batch(x_full_val)                # (B,1,H,W)
            mask_echo_val  = brain_mask_val.expand(-1, N_TARGET_ECHO, -1, -1)     # (B,N_target,H,W)

            val_loss += masked_mse_echo(y_val_hat, x_tgt_val, mask_echo_val).item()
            val_nmse += nmse(y_val_hat * mask_echo_val, x_tgt_val * mask_echo_val).item()


    val_loss /= len(val_loader)
    val_nmse /= len(val_loader)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {train_loss:.6f} | "
          f"Val Loss: {val_loss:.6f} | "
          f"Val NMSE: {val_nmse:.6f}")

    # save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, "best_network3_unet.pth"))

# ------------------------------------------------
# EXPORT A FEW VAL SUBJECTS AS NIFTI
# (similar to network 1, but rmr: we can reconstruct at ANY TEs here)
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
            x_full_slice, _ = ds[i]                         # (N_all, H, W)
            x_full_slice = x_full_slice.to(device).unsqueeze(0)  # (1, N_all, H, W)

            x_in_slice  = x_full_slice[:, input_chan_pos,  :, :]   # (1, N_input, H, W)
            x_tgt_slice = x_full_slice[:, target_chan_pos, :, :]   # (1, N_target, H, W)

            params = model(x_in_slice)                             # (1, 2, H, W)

            # here we reconstruct at TEs_target (you could also define a NEW TE grid)
            y_hat_slice = flash_forward(params, TEs_target)        # (1, N_target, H, W)

            t2s   = torch.abs(params[:, 0]).cpu().numpy().squeeze()
            t1rho = torch.abs(params[:, 1]).cpu().numpy().squeeze()
            recon = y_hat_slice.cpu().numpy().squeeze()           # (N_target, H, W)
            inp   = x_tgt_slice.cpu().numpy().squeeze()           # (N_target, H, W)

            all_t2s.append(t2s)
            all_t1rho.append(t1rho)
            all_recons.append(recon)
            all_inputs.append(inp)

        # stack in z
        t2s_vol   = np.stack(all_t2s, axis=-1)     # (H, W, Z)
        t1rho_vol = np.stack(all_t1rho, axis=-1)   # (H, W, Z)

        recon_vol = np.stack(all_recons, axis=-1)  # (N_target, H, W, Z)
        input_vol = np.stack(all_inputs, axis=-1)  # (N_target, H, W, Z)
        resid_vol = np.abs(input_vol - recon_vol)  # (N_target, H, W, Z)

        # move channels last
        recon_vol = np.transpose(recon_vol, (1, 2, 3, 0))  # (H, W, Z, N_target)
        input_vol = np.transpose(input_vol, (1, 2, 3, 0))
        resid_vol = np.transpose(resid_vol, (1, 2, 3, 0))

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
plt.title("Network 3: TE-mismatched self-supervised training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "loss_curve_network3.png"), dpi=150)
plt.close()

print(f"Training complete. Best val loss: {best_val_loss:.6f}")
