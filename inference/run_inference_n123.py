#!/usr/bin/env python3
"""
Inference entrypoint for FLASH MRI Networks 1–3 for container.

- Net1: UNet(in_channels=N_ECHOES, out_channels=2)
- Net2: UNet(in_channels=N_ECHOES, out_channels=2)
- Net3: UNet(in_channels=N_INPUT_ECHO, out_channels=2)  (typically fewer echoes)

Input format (folder):
  input_dir/
    sub-00001_echo1.npy
    sub-00001_echo2.npy
    sub-00001_echo3.npy
    sub-00001_echo4.npy
    sub-00002_echo1.npy
    ...

Each echo file must be a 3D numpy array: (H, W, Z).

Outputs:
  output/sub-00001/sub-00001_T2star_pred.nii.gz
  output/sub-00001/sub-00001_T1rho_pred.nii.gz

Notes:
- No evaluation/metrics (external user does their own validation).
- Identity affine in NIfTI (because .npy has no header).
"""

import os
import re
import json
import argparse
from typing import Dict, List, Any

import numpy as np
import torch
import nibabel as nib

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.avantika.unet import UNet


SUBJ_ECHO_RE = re.compile(r"^(sub-[^_]+).*_echo(\d+)\.npy$")  # expects sub-XXXX_echoK.npy


def save_nifti(path: str, vol: np.ndarray) -> None:
    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), path)


def load_cases(input_path: str, echo_indices: List[int]) -> List[Dict[str, Any]]:
    """
    input_path: directory OR manifest.json
    returns list of cases:
      { "id": "...", "echo_files": { "1": "...", "2": "...", ... } }
    """
    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if f.endswith(".npy")]
        subj_map: Dict[str, Dict[int, str]] = {}

        for fn in files:
            m = SUBJ_ECHO_RE.match(fn)
            if not m:
                continue
            sid = m.group(1)
            eidx = int(m.group(2))
            subj_map.setdefault(sid, {})[eidx] = os.path.join(input_path, fn)

        cases = []
        for sid, ef in sorted(subj_map.items()):
            missing = [e for e in echo_indices if e not in ef]
            if missing:
                print(f"[WARN] Skipping {sid}: missing echoes {missing}")
                continue
            cases.append({"id": sid, "echo_files": {str(e): ef[e] for e in echo_indices}})
        return cases

    if input_path.lower().endswith(".json"):
        with open(input_path, "r") as f:
            manifest = json.load(f)
        if "cases" not in manifest or not isinstance(manifest["cases"], list):
            raise ValueError("Manifest must contain {'cases': [...]}.")

        cases = []
        for c in manifest["cases"]:
            sid = c.get("id", None)
            ef = c.get("echo_files", {})
            if sid is None or not isinstance(ef, dict):
                raise ValueError("Each case needs keys: 'id' and 'echo_files' dict.")

            missing = [e for e in echo_indices if str(e) not in ef]
            if missing:
                print(f"[WARN] Skipping {sid}: missing echoes {missing} in manifest")
                continue
            cases.append({"id": sid, "echo_files": {str(e): ef[str(e)] for e in echo_indices}})
        return cases

    raise ValueError("--input must be a directory or a .json manifest.")


def load_echo_volume(echo_files: Dict[str, str], echo_indices: List[int]) -> np.ndarray:
    """
    returns echoes shaped (E, H, W, Z)
    """
    vols = []
    for e in echo_indices:
        p = echo_files[str(e)]
        arr = np.load(p)
        if arr.ndim != 3:
            raise ValueError(f"{p} must be (H,W,Z). Got {arr.shape}")
        vols.append(arr.astype(np.float32))
    return np.stack(vols, axis=0)


def load_state_dict_any(checkpoint_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be a state_dict or contain key 'state_dict'.")

    # strip DataParallel prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    return state


def run_unet_inference(model: torch.nn.Module, echoes_ehwz: np.ndarray, device: torch.device) -> (np.ndarray, np.ndarray):
    """
    echoes_ehwz: (E,H,W,Z) -> per-slice model expects (B,E,H,W)
    returns:
      t2s_vol: (H,W,Z)
      t1_vol:  (H,W,Z)
    """
    E, H, W, Z = echoes_ehwz.shape
    t2s_slices = []
    t1_slices = []

    model.eval()
    with torch.no_grad():
        for z in range(Z):
            x = echoes_ehwz[:, :, :, z]  # (E,H,W)
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,E,H,W)
            params = model(x_t)  # (1,2,H,W)

            t2s = torch.abs(params[:, 0]).squeeze(0).cpu().numpy()
            t1  = torch.abs(params[:, 1]).squeeze(0).cpu().numpy()

            t2s_slices.append(t2s)
            t1_slices.append(t1)

    t2s_vol = np.stack(t2s_slices, axis=-1)
    t1_vol  = np.stack(t1_slices, axis=-1)
    return t2s_vol, t1_vol


def main():
    p = argparse.ArgumentParser("FLASH MRI inference (Networks 1–3)")
    p.add_argument("--net", type=int, choices=[1, 2, 3], required=True, help="Which network to run")
    p.add_argument("--input", required=True, help="Input folder of echo npys OR manifest.json")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pth/.pt)")
    p.add_argument("--echo_indices", type=int, nargs="+",
                   help="Echo indices (1-based) to use as INPUT channels. "
                        "For Net1/2 default: 1 2 3 4. For Net3 default: 1 2.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    args = p.parse_args()

    # defaults by net
    if args.echo_indices is None:
        args.echo_indices = [1, 2, 3, 4] if args.net in [1, 2] else [1, 2]

    echo_indices = args.echo_indices
    if any(e <= 0 for e in echo_indices):
        raise ValueError("--echo_indices must be positive 1-based integers.")

    os.makedirs(args.output, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    cases = load_cases(args.input, echo_indices)
    if len(cases) == 0:
        raise RuntimeError(f"No valid cases found in {args.input} for echo_indices={echo_indices}")
    print(f"[INFO] Found {len(cases)} cases.")

    # Model channel counts
    in_ch = len(echo_indices)
    out_ch = 2  # N_PARAMS: [T2*, T1rho] for all nets
    print(f"[INFO] Building UNet(in_channels={in_ch}, out_channels={out_ch}) for Net{args.net}")

    model = UNet(in_channels=in_ch, out_channels=out_ch).to(device)
    state = load_state_dict_any(args.checkpoint, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    for c in cases:
        sid = c["id"]
        print(f"\n[INFO] Running {sid} ...")

        echoes = load_echo_volume(c["echo_files"], echo_indices)  # (E,H,W,Z)
        t2s_vol, t1_vol = run_unet_inference(model, echoes, device)

        out_dir = os.path.join(args.output, sid)
        os.makedirs(out_dir, exist_ok=True)

        save_nifti(os.path.join(out_dir, f"{sid}_T2star_pred.nii.gz"), t2s_vol)
        save_nifti(os.path.join(out_dir, f"{sid}_T1rho_pred.nii.gz"),  t1_vol)

        print(f"[OK] Saved: {out_dir}")

    print("\n[DONE] Inference complete.")


if __name__ == "__main__":
    main()
