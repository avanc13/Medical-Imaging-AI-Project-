#!/usr/bin/env python3
"""
Compare predicted T1rho/T2* maps vs LS maps (LS stored as .npy) AND save
side-by-side images.

Assumes:
  - LS_DIR contains:
        sub-XXXXX_T1rho_ls.npy
        sub-XXXXX_T2star_ls.npy
  - Each experiment has predicted maps in NIfTI:
        <save_dir>/<nifti_subdir>/sub-XXXXX_T1rho_pred.nii.gz
        <save_dir>/<nifti_subdir>/sub-XXXXX_T2star_pred.nii.gz

Metrics are computed:
  - only inside a brain mask derived from echo1
  - and only on voxels where the LS maps lie in a plausible
    physiological range for that parameter.

Outputs:
  - param_metrics_outputs/param_metrics_vs_LS.csv
  - param_metrics_outputs/<exp>_<subject>_param_maps.png
"""

import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.measure import label

# ----------------------------
# configs
# ----------------------------

# LS maps directory (.npy)
LS_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/ground_truth_maps"

# Processed echo1 .npy folder for brain masks
PROCESSED_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data/processed"

OUT_ROOT = "redoing_stuff_12_11/param_metrics_outputs"
os.makedirs(OUT_ROOT, exist_ok=True)

# paths to all the saved validation NIfTI outputs from different experiments
EXPERIMENTS = [
    {
        "name": "net1_4echo_clean",
        "save_dir": "redoing_stuff_12_11/checkpoints_net1_4echoes_with_brainmask",
        "nifti_subdir": "nifti_outputs",
    },
    {
        "name": "net1_4echo_noisy",
        "save_dir": "redoing_stuff_12_11/checkpoints_net1_4echoes_with_brainmask_noisy",
        "nifti_subdir": "nifti_outputs",
    },
    {
        "name": "net1_3echo_clean",
        "save_dir": "redoing_stuff_12_11/checkpoints_net1_3echoes_with_brainmask",
        "nifti_subdir": "nifti_outputs",
    },
    {
        "name": "net1_2echo_clean",
        "save_dir": "redoing_stuff_12_11/checkpoints_net1_2echoes_with_brainmask_echo1_echo4",
        "nifti_subdir": "nifti_outputs",
    },
    {
        "name": "net3_2to4_clean",
        "save_dir": "redoing_stuff_12_11/checkpoints_network3_with_brainmask",
        "nifti_subdir": "nifti_outputs",
    },
    # {
    #     "name": "net3_2to4_noisy",
    #     "save_dir": "checkpoints_noise/checkpoints_network3_noise",
    #     "nifti_subdir": "nifti_outputs",
    # },
    {
        "name": "net2_4echo_LS_supervised",
        "save_dir": "redoing_stuff_12_11/checkpoints_network2_allechoes_with_brainmask",
        "nifti_subdir": "nifti_outputs",
    },
    {
        "name": "net4_4echo_TE_channel",
        "save_dir": "redoing_stuff_12_11/checkpoints_network4_with_brainmask",
        "nifti_subdir": "nifti_outputs",
    },
]

MASK_PERCENTILE = 60.0

# Physiologic / plausible ranges (seconds)
T1RHO_PLAUSIBLE_MIN = 0.2
T1RHO_PLAUSIBLE_MAX = 3.0

# T2* ~20–180 ms as plausible brain range
T2_PLAUSIBLE_MIN = 0.02
T2_PLAUSIBLE_MAX = 0.18

# Visualization ranges (shared across LS/pred)
T1RHO_VMIN, T1RHO_VMAX = 0.0, 3.0
T2_VMIN,    T2_VMAX    = 0.0, 0.18  # you can bump to 0.20 if you want a bit more headroom

MAX_VIS_PER_EXP = 1  # one figure per experiment


# ----------------------------
# HELPERS
# ----------------------------

def build_brain_mask(subject_id, H, W, Z):
    echo1_path = os.path.join(PROCESSED_DIR, f"{subject_id}_echo1.npy")
    if not os.path.exists(echo1_path):
        raise FileNotFoundError(f"Echo1 .npy not found for {subject_id}: {echo1_path}")

    echo1 = np.load(echo1_path)
    if echo1.shape != (H, W, Z):
        raise ValueError(
            f"Shape mismatch between echo1 {echo1.shape} and params {(H, W, Z)} "
            f"for {subject_id}"
        )

    nonzero_vals = echo1[echo1 > 0]
    if nonzero_vals.size == 0:
        raise ValueError(f"Echo1 for {subject_id} is all zeros; cannot build mask.")

    thr = np.percentile(nonzero_vals, MASK_PERCENTILE)
    mask = echo1 > thr

    mask = binary_closing(mask, structure=np.ones((3, 3, 3)))
    mask = binary_fill_holes(mask)

    label_map = label(mask)
    if label_map.max() == 0:
        raise ValueError(
            f"No connected components in mask for {subject_id}; threshold {thr:.5f}"
        )
    largest_cc = np.argmax(np.bincount(label_map.flat)[1:]) + 1
    mask = (label_map == largest_cc)

    return mask


def safe_corr(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def save_param_figure(exp_name, subject_id, ls_t1, pred_t1, ls_t2, pred_t2, mask):
    """Side-by-side maps for one subject & experiment, masked + shared scale."""
    H, W, Z = ls_t1.shape
    mid_slice = Z // 2

    m = mask[:, :, mid_slice]

    ls_t1_s   = np.where(m, ls_t1[:, :, mid_slice], np.nan)
    pred_t1_s = np.where(m, pred_t1[:, :, mid_slice], np.nan)
    diff_t1_s = np.clip(np.abs(pred_t1_s - ls_t1_s), 0.0, T1RHO_VMAX)

    ls_t2_s   = np.where(m, ls_t2[:, :, mid_slice], np.nan)
    pred_t2_s = np.where(m, pred_t2[:, :, mid_slice], np.nan)
    diff_t2_s = np.clip(np.abs(pred_t2_s - ls_t2_s), 0.0, T2_VMAX)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    # Row 1: T1rho
    im0 = axes[0, 0].imshow(np.ma.masked_invalid(ls_t1_s),
                            vmin=T1RHO_VMIN, vmax=T1RHO_VMAX, cmap="viridis")
    axes[0, 0].set_title("T1ρ LS"); axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(np.ma.masked_invalid(pred_t1_s),
                            vmin=T1RHO_VMIN, vmax=T1RHO_VMAX, cmap="viridis")
    axes[0, 1].set_title("T1ρ pred"); axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(np.ma.masked_invalid(diff_t1_s),
                            vmin=0.0, vmax=T1RHO_VMAX, cmap="magma")
    axes[0, 2].set_title("|T1ρ diff|"); axes[0, 2].axis("off")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 2: T2*
    im3 = axes[1, 0].imshow(np.ma.masked_invalid(ls_t2_s),
                            vmin=T2_VMIN, vmax=T2_VMAX, cmap="viridis")
    axes[1, 0].set_title("T2* LS"); axes[1, 0].axis("off")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(np.ma.masked_invalid(pred_t2_s),
                            vmin=T2_VMIN, vmax=T2_VMAX, cmap="viridis")
    axes[1, 1].set_title("T2* pred"); axes[1, 1].axis("off")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im5 = axes[1, 2].imshow(np.ma.masked_invalid(diff_t2_s),
                            vmin=0.0, vmax=T2_VMAX, cmap="magma")
    axes[1, 2].set_title("|T2* diff|"); axes[1, 2].axis("off")
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle(f"{exp_name} – {subject_id} (mid-slice, masked)", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = os.path.join(OUT_ROOT, f"{exp_name}_{subject_id}_param_maps.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved param map figure: {out_path}")


# ----------------------------
# MAIN
# ----------------------------

# Discover subjects from LS T2* .npy files
ls_t2_files = sorted(glob.glob(os.path.join(LS_DIR, "*_T2star.npy")))
subjects = [
    os.path.basename(p).replace("_T2star.npy", "")
    for p in ls_t2_files
]

print(f"Found {len(subjects)} subjects in LS_DIR")

rows = []

for exp in EXPERIMENTS:
    exp_name  = exp["name"]
    save_dir  = exp["save_dir"]
    nifti_dir = os.path.join(save_dir, exp["nifti_subdir"])

    print(f"\n=== Experiment: {exp_name} ===")
    print(f"NIfTI dir: {nifti_dir}")

    vis_count = 0

    for subject_id in subjects:
        ls_t1_path = os.path.join(LS_DIR, f"{subject_id}_T1p.npy")
        ls_t2_path = os.path.join(LS_DIR, f"{subject_id}_T2star.npy")
        if not (os.path.exists(ls_t1_path) and os.path.exists(ls_t2_path)):
            continue

        # predicted maps still assumed to be NIfTI
        pred_t1_path = os.path.join(nifti_dir, f"{subject_id}_T1rho_pred.nii.gz")
        pred_t2_path = os.path.join(nifti_dir, f"{subject_id}_T2star_pred.nii.gz")
        if not (os.path.exists(pred_t1_path) and os.path.exists(pred_t2_path)):
            continue

        # Load LS from .npy
        ls_t1 = np.load(ls_t1_path)  # (H,W,Z)
        ls_t2 = np.load(ls_t2_path)

        # Load preds from NIfTI
        pred_t1 = nib.load(pred_t1_path).get_fdata()
        pred_t2 = nib.load(pred_t2_path).get_fdata()

        if ls_t1.shape != ls_t2.shape or ls_t1.shape != pred_t1.shape or ls_t1.shape != pred_t2.shape:
            raise ValueError(f"Shape mismatch in param maps for {subject_id} in {exp_name}")

        H, W, Z = ls_t1.shape
        mask = build_brain_mask(subject_id, H, W, Z)
        mask_flat = mask.reshape(-1)

        # ---- T1rho metrics (brain + plausible LS range) ----
        ls_t1_flat_all   = ls_t1.reshape(-1)[mask_flat]
        pred_t1_flat_all = pred_t1.reshape(-1)[mask_flat]

        plausible_t1_mask = (
            (ls_t1_flat_all >= T1RHO_PLAUSIBLE_MIN) &
            (ls_t1_flat_all <= T1RHO_PLAUSIBLE_MAX)
        )

        ls_t1_flat   = ls_t1_flat_all[plausible_t1_mask]
        pred_t1_flat = pred_t1_flat_all[plausible_t1_mask]

        if ls_t1_flat.size == 0:
            mae_t1  = np.nan
            corr_t1 = np.nan
            n_t1    = 0
        else:
            diff_t1 = pred_t1_flat - ls_t1_flat
            mae_t1  = float(np.mean(np.abs(diff_t1)))
            corr_t1 = safe_corr(pred_t1_flat, ls_t1_flat)
            n_t1    = int(ls_t1_flat.size)

        # ---- T2* metrics (brain + plausible LS range) ----
        ls_t2_flat_all   = ls_t2.reshape(-1)[mask_flat]
        pred_t2_flat_all = pred_t2.reshape(-1)[mask_flat]

        plausible_t2_mask = (
            (ls_t2_flat_all >= T2_PLAUSIBLE_MIN) &
            (ls_t2_flat_all <= T2_PLAUSIBLE_MAX)
        )

        ls_t2_flat   = ls_t2_flat_all[plausible_t2_mask]
        pred_t2_flat = pred_t2_flat_all[plausible_t2_mask]

        if ls_t2_flat.size == 0:
            mae_t2  = np.nan
            corr_t2 = np.nan
            n_t2    = 0
        else:
            diff_t2 = pred_t2_flat - ls_t2_flat
            mae_t2  = float(np.mean(np.abs(diff_t2)))
            corr_t2 = safe_corr(pred_t2_flat, ls_t2_flat)
            n_t2    = int(ls_t2_flat.size)

        rows.append({
            "experiment": exp_name,
            "subject": subject_id,
            "mae_T1rho": mae_t1,
            "corr_T1rho": corr_t1,
            "nvox_T1rho": n_t1,
            "mae_T2star": mae_t2,
            "corr_T2star": corr_t2,
            "nvox_T2star": n_t2,
        })

        # one visualization per experiment
        if vis_count < MAX_VIS_PER_EXP:
            save_param_figure(exp_name, subject_id, ls_t1, pred_t1, ls_t2, pred_t2, mask)
            vis_count += 1

# ---- save CSV + summary ----
if rows:
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_ROOT, "param_metrics_vs_LS_network1masked.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved param metrics CSV: {csv_path}")

    summary = (
        df.groupby("experiment")[["mae_T1rho", "corr_T1rho", "mae_T2star", "corr_T2star"]]
        .mean()
        .reset_index()
    )
    print("\nExperiment-level mean metrics "
          "(brain mask, LS in physiologic range only):")
    print(summary)
else:
    print("\nNo rows collected; check LS_DIR, EXPERIMENTS, and file naming.")
