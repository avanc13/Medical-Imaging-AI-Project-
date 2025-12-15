#!/usr/bin/env python3
"""
Network 4 inference (TE-conditioned, self-supervised model).

Given a folder of processed echo .npy files (or a subject id + data_dir),
loads a trained Net4 checkpoint, predicts per-slice params by averaging
across echoes, reconstructs echoes via physics model, and saves NIfTI outputs.

Outputs:
  - <subj>_T2star_pred.nii.gz         (H,W,Z)
  - <subj>_T1rho_pred.nii.gz          (H,W,Z)
  - <subj>_reconstructed_echoes.nii.gz (H,W,Z,N_echoes)
  - <subj>_residuals.nii.gz            (H,W,Z,N_echoes)
"""

import os, sys, argparse
import numpy as np
import torch
import nibabel as nib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataloaders.flash_dataset import FlashMRIDataset
from models.avantika.unet import UNet


def flash_forward_single_te(params, te_vals):
    """
    params: (B,2,H,W) -> [T2*, T1rho]
    te_vals: (B,) seconds
    returns: (B,1,H,W)
    """
    T2s   = torch.abs(params[:, 0:1]) + 1e-3
    T1rho = torch.abs(params[:, 1:2]) + 1e-3
    te = te_vals.view(-1, 1, 1, 1)
    return T1rho * torch.exp(-te / T2s)


def parse_args():
    p = argparse.ArgumentParser("FLASH MRI inference (Network 4)")
    p.add_argument("--data_dir", required=True,
                   help="Processed data directory containing sub-*_echo*.npy")
    p.add_argument("--subject", required=True,
                   help="Subject ID like sub-19979")
    p.add_argument("--checkpoint", required=True,
                   help="Path to Net4 checkpoint (.pth/.pt)")
    p.add_argument("--output", required=True,
                   help="Output directory for NIfTI files")
    p.add_argument("--echo_indices", nargs="+", type=int, default=[1,2,3,4],
                   help="Echo indices to use (1-based), default 1 2 3 4")
    p.add_argument("--tes", nargs="+", type=float, default=[0.012, 0.028, 0.044, 0.060],
                   help="Echo times in seconds; must align with echo_indices order")
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    assert len(args.echo_indices) == len(args.tes), "echo_indices and tes must be same length"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs(args.output, exist_ok=True)

    tes = torch.tensor(args.tes, dtype=torch.float32, device=device)  # (N_echoes,)

    # dataset yields per-slice: x_full_slice (N_echoes,H,W)
    ds = FlashMRIDataset([args.subject], args.data_dir,
                         echo_indices=args.echo_indices, mode="val")

    # model: input is [echo, te_map] => 2 channels; output 2 params
    model = UNet(in_channels=2, out_channels=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # handle either raw state_dict or wrapped dicts
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    all_t2s, all_t1rho = [], []
    all_recons, all_inputs = [], []

    with torch.no_grad():
        for i in range(len(ds)):
            x_full_slice, _ = ds[i]  # (N_echoes,H,W)
            x_full_slice = x_full_slice.unsqueeze(0).to(device)  # (1,N_echoes,H,W)
            _, Ce, H, W = x_full_slice.shape

            # estimate params from each echo, then average
            params_per_echo = []
            for e in range(Ce):
                x_echo = x_full_slice[:, e:e+1]  # (1,1,H,W)
                te_val = tes[e].view(1)          # (1,)

                te_map = te_val.view(1,1,1,1).expand(-1,1,H,W)  # (1,1,H,W)
                x_in   = torch.cat([x_echo, te_map], dim=1)     # (1,2,H,W)

                params = model(x_in)  # (1,2,H,W)
                params_per_echo.append(params)

            params_mean = torch.stack(params_per_echo, dim=0).mean(dim=0)  # (1,2,H,W)

            t2s_map   = torch.abs(params_mean[:,0]).cpu().numpy().squeeze()   # (H,W)
            t1rho_map = torch.abs(params_mean[:,1]).cpu().numpy().squeeze()   # (H,W)

            recons = []
            inputs = []
            for e in range(Ce):
                te_val = tes[e].view(1)  # (1,)
                y_hat  = flash_forward_single_te(params_mean, te_val)  # (1,1,H,W)
                recons.append(y_hat.cpu().numpy().squeeze())
                inputs.append(x_full_slice[:, e].cpu().numpy().squeeze())

            all_t2s.append(t2s_map)
            all_t1rho.append(t1rho_map)
            all_recons.append(np.stack(recons, axis=0))  # (Ce,H,W)
            all_inputs.append(np.stack(inputs, axis=0))  # (Ce,H,W)

    # stack over slices => volumes
    t2s_vol   = np.stack(all_t2s, axis=-1)          # (H,W,Z)
    t1rho_vol = np.stack(all_t1rho, axis=-1)        # (H,W,Z)

    recon_vol = np.stack(all_recons, axis=-1)       # (Ce,H,W,Z)
    input_vol = np.stack(all_inputs, axis=-1)       # (Ce,H,W,Z)
    resid_vol = np.abs(input_vol - recon_vol)       # (Ce,H,W,Z)

    # echoes last for NIfTI convenience
    recon_vol = np.transpose(recon_vol, (1,2,3,0))  # (H,W,Z,Ce)
    resid_vol = np.transpose(resid_vol, (1,2,3,0))  # (H,W,Z,Ce)

    affine = np.eye(4)
    nib.save(nib.Nifti1Image(t2s_vol.astype(np.float32), affine),
             os.path.join(args.output, f"{args.subject}_T2star_pred.nii.gz"))
    nib.save(nib.Nifti1Image(t1rho_vol.astype(np.float32), affine),
             os.path.join(args.output, f"{args.subject}_T1rho_pred.nii.gz"))
    nib.save(nib.Nifti1Image(recon_vol.astype(np.float32), affine),
             os.path.join(args.output, f"{args.subject}_reconstructed_echoes.nii.gz"))
    nib.save(nib.Nifti1Image(resid_vol.astype(np.float32), affine),
             os.path.join(args.output, f"{args.subject}_residuals.nii.gz"))

    print(f"[Net4] Saved outputs to: {args.output}")


if __name__ == "__main__":
    main()
