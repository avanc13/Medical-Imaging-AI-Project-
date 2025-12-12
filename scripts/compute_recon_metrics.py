#!/usr/bin/env python3
"""
Compute brain-masked reconstruction metrics for multiple experiments.

For each experiment (network + echo/noise setting), this script:
  - Finds *_residuals.nii.gz in the given NIfTI output folder
  - Builds a brain mask from echo1 .npy (same for all networks)
  - Computes per-subject, per-echo:
        * mean |residual|
        * MSE
  - Saves a single CSV with all experiments
  - Saves one bar plot per experiment (overall mean MSE per echo)

You can directly use the CSV for tables and the PNGs for slides.
"""

import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.measure import label

# ----------------------------
# CONFIG
# ----------------------------

# Folder with processed echo .npy files (clean or noisy) used for MASKING.
# Use the same PROCESSED_DIR for all experiments so the ROI is identical.
PROCESSED_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data/processed"

# Where to put summary outputs (CSV + plots)
OUT_ROOT = "redoing_stuff_12_11/recon_metrics_outputs"
os.makedirs(OUT_ROOT, exist_ok=True)

# Experiments to evaluate.
# Edit this list to match your checkpoint dirs.
# Example:
#   name: label used in CSV / plots
#   save_dir: where that network's checkpoints live
#   nifti_subdir: subfolder under save_dir with residual NIfTIs
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


# Brain mask threshold percentile (on echo1 nonzero values)
MASK_PERCENTILE = 60.0  # tweak (50–60) if needed

# ----------------------------
# HELPER: build brain mask from echo1 .npy
# ----------------------------

def build_brain_mask_and_echo1(subject_id, H, W, Z):
    echo1_path = os.path.join(PROCESSED_DIR, f"{subject_id}_echo1.npy")
    if not os.path.exists(echo1_path):
        raise FileNotFoundError(f"Echo1 .npy not found for {subject_id}: {echo1_path}")

    echo1 = np.load(echo1_path)
    if echo1.shape != (H, W, Z):
        raise ValueError(
            f"Shape mismatch between echo1 {echo1.shape} and residuals {(H, W, Z)} "
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

    return mask, echo1


# ----------------------------
# MAIN
# ----------------------------

all_rows = []

for exp in EXPERIMENTS:
    exp_name = exp["name"]
    save_dir = exp["save_dir"]
    nifti_dir = os.path.join(save_dir, exp["nifti_subdir"])

    print(f"\n=== Experiment: {exp_name} ===")
    print(f"NIfTI dir: {nifti_dir}")

    resid_files = sorted(glob.glob(os.path.join(nifti_dir, "*_residuals.nii.gz")))
    if not resid_files:
        print(f"  WARNING: no residuals found in {nifti_dir}, skipping.")
        continue

    # To accumulate per-echo means per experiment (for bar plot)
    exp_echo_mse = {}        # echo_idx -> list of MSE across subjects
    exp_echo_mean_abs = {}   # echo_idx -> list of mean|resid| across subjects
    example_image_done = False

    for resid_path in resid_files:
        subject_id = os.path.basename(resid_path).replace("_residuals.nii.gz", "")
        print(f"  Subject: {subject_id}")

        # Load residuals
        resid = nib.load(resid_path).get_fdata()
        if resid.ndim != 4:
            raise ValueError(
                f"Unexpected residual ndim={resid.ndim} for {subject_id}"
            )

        H, W, Z, C = resid.shape

        # Fix channels-first if needed
        if C <= 8 and H > C and W > C and Z > C:
            pass  # already (H, W, Z, C)
        elif H <= 8 and C > H:
            resid = np.transpose(resid, (1, 2, 3, 0))
            H, W, Z, C = resid.shape
            print(f"    Transposed residuals to (H, W, Z, C) = {resid.shape}")
        else:
            print(f"    WARNING: unusual residual shape {resid.shape}, proceeding.")

        # Build / get brain mask
        #mask = build_brain_mask(subject_id, H, W, Z)  # (H, W, Z)
        mask, echo1 = build_brain_mask_and_echo1(subject_id, H, W, Z)


        resid_abs = np.abs(resid)        # |residual|
        resid_sq = resid ** 2            # residual^2

        # Flatten with mask
        resid_abs_flat = resid_abs.reshape(-1, C)  # (N_vox, C)
        resid_sq_flat = resid_sq.reshape(-1, C)
        mask_flat = mask.reshape(-1)

        brain_abs = resid_abs_flat[mask_flat]  # (N_brain, C)
        brain_sq = resid_sq_flat[mask_flat]

        if brain_abs.size == 0:
            raise ValueError(f"Empty brain mask for {subject_id} in {exp_name}")

        mean_abs_per_echo = brain_abs.mean(axis=0)    # (C,)
        mse_per_echo = brain_sq.mean(axis=0)          # (C,)

        # Record per-echo metrics for this subject
        for echo_idx in range(C):
            echo_num = echo_idx + 1
            all_rows.append({
                "experiment": exp_name,
                "subject": subject_id,
                "echo_idx": echo_num,
                "mean_abs_resid": float(mean_abs_per_echo[echo_idx]),
                "mse": float(mse_per_echo[echo_idx]),
            })

            exp_echo_mse.setdefault(echo_num, []).append(mse_per_echo[echo_idx])
            exp_echo_mean_abs.setdefault(echo_num, []).append(mean_abs_per_echo[echo_idx])

        if not example_image_done:
            mid_slice = Z // 2
            mean_resid_map = resid_abs.mean(axis=-1)  # (H,W,Z)
            mean_resid_map_masked = mean_resid_map.copy()
            mean_resid_map_masked[~mask] = 0.0

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(echo1[:, :, mid_slice], cmap="gray")
            plt.title(f"{subject_id} echo1")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(mean_resid_map_masked[:, :, mid_slice], cmap="hot")
            plt.title(f"{subject_id} mean |resid| (brain)")
            plt.axis("off")

            plt.tight_layout()
            out_img = os.path.join(
                OUT_ROOT, f"{exp_name}_{subject_id}_echo1_and_residual.png"
            )
            plt.savefig(out_img, dpi=150)
            plt.close()
            print(f"  Saved example echo1/residual image: {out_img}")
            example_image_done = True

    # Make per-experiment bar plot (overall MSE per echo)
    if exp_echo_mse:
        echo_nums = sorted(exp_echo_mse.keys())
        mean_mse = [np.mean(exp_echo_mse[e]) for e in echo_nums]

        plt.figure(figsize=(5, 4))
        plt.bar(echo_nums, mean_mse)
        plt.xticks(echo_nums, [str(e) for e in echo_nums])
        plt.xlabel("Echo index")
        plt.ylabel("Mean MSE (brain only)")
        plt.title(f"{exp_name}: overall MSE per echo")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(OUT_ROOT, f"{exp_name}_MSE_per_echo.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved MSE bar plot: {plot_path}")

# Save full CSV
if all_rows:
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_ROOT, "recon_metrics_brain_only.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved reconstruction metrics CSV: {csv_path}")

    # Optional: print quick overall summary
    summary = (
        df.groupby(["experiment", "echo_idx"])[["mean_abs_resid", "mse"]]
        .mean()
        .reset_index()
    )
    print("\nOverall mean metrics (brain only):")
    print(summary)
else:
    print("\nNo rows collected; check your EXPERIMENTS config and paths.")
