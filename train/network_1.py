#!/usr/bin/env python3
"""
Self-supervised training script (Network 1) for multi-echo FLASH MRI.
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

from dataloaders.flash_dataset import FlashMRIDataset
from models.avantika.unet import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data/processed"

# get all subjects
subjects = sorted(set([os.path.basename(f).split("_echo")[0] for f in glob.glob(os.path.join(DATA_DIR, "sub-*_echo1.npy"))]))
print(f"Found {len(subjects)} subjects total.")

# remove corrupted or incomplete subjects
subjects = [s for s in subjects if s != "sub-04620"]
print(f"Remaining {len(subjects)} subjects after removing corrupted entries.")

# split into train and validation
train_subj, val_subj = train_test_split(subjects, test_size=0.125, random_state=42)
print(f"Training on {len(train_subj)} subjects, validating on {len(val_subj)} subjects.")

TRAIN_SUBJECTS = train_subj
VAL_SUBJECTS = val_subj

N_ECHOES = 4
N_PARAMS = 2   # [T2*, t1p] 
BATCH_SIZE = 4
NUM_EPOCHS = 50
LR = 1e-4
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# echo times (in seconds) from the json metadata
TEs = torch.tensor([0.012, 0.028, 0.044, 0.06]).to(device)




# PHYSICS MODEL
def flash_forward(params, TEs):
    """
    FLASH MRI forward model:
        y(TE) = (T1rho) * exp(-TE / T2*)
    params: (B, 2, H, W) → [T2*, T1rho]
    TEs: (N_echoes,)
    Returns: (B, N_echoes, H, W)
    """
    T2s   = torch.abs(params[:, 0:1, :, :]) + 1e-3  #add abs to avoid zero division
    T1rho = torch.abs(params[:, 1:2, :, :]) + 1e-3 #add abs to avoid negative intensities
    y_hat = T1rho * torch.exp(-TEs.view(1, -1, 1, 1) / T2s)
    return y_hat


# DATASET / DATALOADERS
# gonna allow us to vary echo indices later 
train_ds = FlashMRIDataset(TRAIN_SUBJECTS, DATA_DIR, echo_indices=list(range(1, N_ECHOES + 1)), mode="train")
val_ds   = FlashMRIDataset(VAL_SUBJECTS,   DATA_DIR, echo_indices=list(range(1, N_ECHOES + 1)), mode="val")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# MODEL / LOSS / OPTIMIZER

model = UNet(in_channels=N_ECHOES, out_channels=N_PARAMS).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

# tracking
train_losses, val_losses = [], []
best_val_loss = float("inf")

print("\n===  verifying shapes ===")
x, sid = next(iter(train_loader))
print(f"batch from subjects: {sid[:4]}")
print(f"input shape: {x.shape}")  # should be [B, 4, 64, 64]

x = x.to(device)
params_pred = model(x)
print(f"Predicted params shape: {params_pred.shape}")  # [B, 2, 64, 64]

y_hat = flash_forward(params_pred, TEs)
print(f"Reconstructed echoes shape: {y_hat.shape}")  # [B, 4, 64, 64]

print("make sure above are consistent.\n")


# ------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_idx, (x, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)):
        x = x.to(device)

        #  parameters
        params_pred = model(x)

        # apply physics model on parameters to get predicted echoes
        y_hat = flash_forward(params_pred, TEs)

        if epoch == 0 and batch_idx == 0:
            # print some stats for the first batch
            print("Input range:", x.min().item(), x.max().item())
            print("Predicted T2* range:", params_pred[:,0].min().item(), params_pred[:,0].max().item())
            print("Predicted T1rho range:", params_pred[:,1].min().item(), params_pred[:,1].max().item())
            print("y_hat range:", y_hat.min().item(), y_hat.max().item())


        # self-supervised reconstruction loss: loss between predicted echoes and input echoes
        loss = criterion(y_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # ------------------------------------------------
    # VALIDATION
    # ------------------------------------------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val, _ in val_loader:
            x_val = x_val.to(device)
            params_val = model(x_val)
            y_val_hat = flash_forward(params_val, TEs)
            val_loss += criterion(y_val_hat, x_val).item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # save best checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_network1_unet.pth"))


# saving some of the val outputs as NIfTI for visualization
# --- make output dir ---
SAVE_NIFTI_DIR = os.path.join(SAVE_DIR, "nifti_outputs")
os.makedirs(SAVE_NIFTI_DIR, exist_ok=True)

model.eval()
with torch.no_grad():
    for subj in VAL_SUBJECTS[:3]:  # export a few validation subjects
        # collect all slices for that subject
        ds = FlashMRIDataset([subj], DATA_DIR, echo_indices=list(range(1, N_ECHOES + 1)), mode="val")
        all_t2s, all_t1rho = [], []
        all_recons, all_inputs = [], []

        for i in range(len(ds)):
            x, _ = ds[i]                   # (4, H, W)
            x = x.unsqueeze(0).to(device)  # add batch dim

            params = model(x)              # (1, 2, H, W)
            y_hat = flash_forward(params, TEs)  # (1, 4, H, W)

            t2s   = torch.abs(params[:,0]).cpu().numpy().squeeze()
            t1rho = torch.abs(params[:,1]).cpu().numpy().squeeze()
            recon = y_hat.cpu().numpy().squeeze()
            inp   = x.cpu().numpy().squeeze()

            all_t2s.append(t2s)
            all_t1rho.append(t1rho)
            all_recons.append(recon)
            all_inputs.append(inp)

        # stack slices in z-axis
        t2s_vol   = np.stack(all_t2s, axis=-1)
        t1rho_vol = np.stack(all_t1rho, axis=-1)
        recon_vol = np.stack(all_recons, axis=-1)
        input_vol = np.stack(all_inputs, axis=-1)
        resid_vol = np.abs(input_vol - recon_vol)

        # save as NIfTI
        affine = np.eye(4)  # identity (no spatial metadata)
        nib.save(nib.Nifti1Image(t2s_vol, affine),   os.path.join(SAVE_NIFTI_DIR, f"{subj}_T2star_pred.nii.gz"))
        nib.save(nib.Nifti1Image(t1rho_vol, affine), os.path.join(SAVE_NIFTI_DIR, f"{subj}_T1rho_pred.nii.gz"))
        nib.save(nib.Nifti1Image(recon_vol, affine), os.path.join(SAVE_NIFTI_DIR, f"{subj}_reconstructed_echoes.nii.gz"))
        nib.save(nib.Nifti1Image(resid_vol, affine), os.path.join(SAVE_NIFTI_DIR, f"{subj}_residuals.nii.gz"))

        print(f"Saved NIfTI outputs for {subj} → {SAVE_NIFTI_DIR}")


# ------------------------------------------------
# PLOTS
# ------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_losses, label="Validation Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Self-supervised Training (Network 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "loss_curve_network1.png"), dpi=150)
plt.show()

print(f"Training complete. Best val loss: {best_val_loss:.6f}")
