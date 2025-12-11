#!/usr/bin/env python3
"""
Inference script for Network 1 (self-supervised FLASH MRI).

- Loads a trained UNet checkpoint.
- Runs inference on ALL subjects it finds in --data_dir.
- Expects files like: sub-XXXXX_echo1.npy, sub-XXXXX_echo2.npy, ...
- Saves predicted T2* and T1rho volumes as NIfTI in --save_dir.

This is the script the container will run (no training, no plotting).
"""

import os
import sys
import glob
import argparse

import numpy as np
import torch
import nibabel as nib

# make sure repo root is on path if run from container or elsewhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.avantika.unet import UNet
from dataloaders.flash_dataset import FlashMRIDataset


def flash_forward(params, TEs):
    """
    FLASH MRI forward model:
        y(TE) = (T1rho) * exp(-TE / T2*)
    params: (B, 2, H, W) → [T2*, T1rho]
    TEs: (N_echoes,) torch tensor
    Returns: (B, N_echoes, H, W)
    """
    T2s   = torch.abs(params[:, 0:1, :, :]) + 1e-3
    T1rho = torch.abs(params[:, 1:2, :, :]) + 1e-3
    y_hat = T1rho * torch.exp(-TEs.view(1, -1, 1, 1) / T2s)
    return y_hat


def find_subjects(data_dir, echo_indices):
    """
    Find all subjects in data_dir that have ALL requested echo files.

    We first find all sub-*_echo1.npy, then keep only those subjects
    that also have sub-*_echoK.npy for every K in echo_indices.
    """
    # find candidates by echo1
    echo1_files = glob.glob(os.path.join(data_dir, "sub-*_*echo1.npy"))
    if len(echo1_files) == 0:
        echo1_files = glob.glob(os.path.join(data_dir, "sub-*_echo1.npy"))

    candidates = sorted(set(os.path.basename(f).split("_echo")[0] for f in echo1_files))

    valid_subjects = []
    for subj in candidates:
        ok = True
        for e in echo_indices:
            path = os.path.join(data_dir, f"{subj}_echo{e}.npy")
            if not os.path.exists(path):
                print(f"[WARN] Skipping {subj}: missing {path}")
                ok = False
                break
        if ok:
            valid_subjects.append(subj)

    return valid_subjects



def main():
    parser = argparse.ArgumentParser(description="Network 1 inference (FLASH MRI)")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with processed echo .npy files (sub-*_echo*.npy).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained Network 1 checkpoint (.pth).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save predicted parameter maps (NIfTI).",
    )
    parser.add_argument(
        "--echo_indices",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Echo indices to use (1-based). Default: 1 2 3 4",
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # echo times (seconds) – must match what you used in training
    TEs_all = torch.tensor([0.012, 0.028, 0.044, 0.060], dtype=torch.float32, device=device)
    # convert 1-based echo indices to 0-based
    echo_indices_zero = [i - 1 for i in args.echo_indices]
    TEs = TEs_all[echo_indices_zero]
    n_echoes = len(args.echo_indices)

    # ---- load model ----
    print(f"Loading Network 1 UNet from {args.checkpoint_path}")
    model = UNet(in_channels=n_echoes, out_channels=2).to(device)
    state = torch.load(args.checkpoint_path, map_location=device)
    # allow for state dicts saved with extra keys (if needed)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    # ---- find subjects ----
    subjects = find_subjects(args.data_dir, args.echo_indices)
    if len(subjects) == 0:
        raise RuntimeError(f"No subjects found in {args.data_dir} with pattern sub-*_echo1.npy")

    print(f"Found {len(subjects)} subjects for inference.")
    print("Subjects:", subjects)

    # ---- run inference per subject ----
    for subj in subjects:
        print(f"\nRunning inference for {subj} ...")

        # dataset for this subject only; mode "val" or "inference" doesn't really matter
        ds = FlashMRIDataset([subj], args.data_dir, echo_indices=args.echo_indices, mode="val")

        all_t2s   = []
        all_t1rho = []

        # we also optionally save reconstructed echoes and input for debugging if needed
        all_recons = []
        all_inputs = []

        with torch.no_grad():
            for i in range(len(ds)):
                x, _ = ds[i]              # (C, H, W)
                x = x.unsqueeze(0).to(device)  # (1, C, H, W)

                params = model(x)              # (1, 2, H, W)
                y_hat  = flash_forward(params, TEs)  # (1, n_echoes, H, W)

                t2s   = torch.abs(params[:, 0]).cpu().numpy().squeeze()
                t1rho = torch.abs(params[:, 1]).cpu().numpy().squeeze()
                recon = y_hat.cpu().numpy().squeeze()
                inp   = x.cpu().numpy().squeeze()

                all_t2s.append(t2s)
                all_t1rho.append(t1rho)
                all_recons.append(recon)
                all_inputs.append(inp)

        # stack slices along z-axis
        t2s_vol   = np.stack(all_t2s, axis=-1)        # (H, W, Z)
        t1rho_vol = np.stack(all_t1rho, axis=-1)
        recon_vol = np.stack(all_recons, axis=-1)     # (n_echoes, H, W, Z)
        input_vol = np.stack(all_inputs, axis=-1)     # (n_echoes, H, W, Z)

        # rearrange to (H, W, Z, C) for NIfTI
        recon_vol = np.transpose(recon_vol, (1, 2, 3, 0))
        input_vol = np.transpose(input_vol, (1, 2, 3, 0))
        resid_vol = np.abs(input_vol - recon_vol)

        affine = np.eye(4)

        out_dir_subj = os.path.join(args.save_dir, subj)
        os.makedirs(out_dir_subj, exist_ok=True)

        nib.save(nib.Nifti1Image(t2s_vol, affine),   os.path.join(out_dir_subj, f"{subj}_T2star_pred.nii.gz"))
        nib.save(nib.Nifti1Image(t1rho_vol, affine), os.path.join(out_dir_subj, f"{subj}_T1rho_pred.nii.gz"))
        nib.save(nib.Nifti1Image(recon_vol, affine), os.path.join(out_dir_subj, f"{subj}_reconstructed_echoes.nii.gz"))
        nib.save(nib.Nifti1Image(resid_vol, affine), os.path.join(out_dir_subj, f"{subj}_residuals.nii.gz"))

        print(f"Saved predictions for {subj} to {out_dir_subj}")

    print("\nInference complete for all subjects.")


if __name__ == "__main__":
    main()
