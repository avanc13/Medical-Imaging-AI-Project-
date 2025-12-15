#!/usr/bin/env python3
"""
Compute percentage of predicted voxels within physiologic range
inside a brain mask, per experiment and subject.
"""

import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.measure import label

# ----------------------------
# CONFIG
# ----------------------------

PROCESSED_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data/processed"

OUT_DIR = "../models/varsha/varsha_data_temp"
os.makedirs(OUT_DIR, exist_ok=True)

MASK_PERCENTILE = 60.0

# Physiologic ranges (seconds)
T1RHO_MIN, T1RHO_MAX = 0.2, 3.0
T2_MIN,    T2_MAX    = 0.02, 0.18

EXPERIMENTS = [
    {
        "name": "net2_4echo_wbm",
        "save_dir": "../redoing_stuff_12_12/checkpoints_network2_allechoes_with_brainmask",
        "nifti_subdir": "nifti_outputs",
    },
    {
        "name": "net2_3echo_wbm",
        "save_dir": "../redoing_stuff_12_14/checkpoints_network2_3echoes_with_brainmask",
        "nifti_subdir": "nifti_outputs",
    },
    {
        "name": "net2_2echo_wbm",
        "save_dir": "../redoing_stuff_12_14/checkpoints_network2_2echoes_with_brainmask",
        "nifti_subdir": "nifti_outputs",
    },
        {
        "name": "net2_1echo_wbm",
        "save_dir": "../redoing_stuff_12_14/checkpoints_network2_1echoes_with_brainmask",
        "nifti_subdir": "nifti_outputs",
    },
            {
        "name": "net4_film_wbm",
        "save_dir": "../redoing_stuff_12_14/network4_film",
        "nifti_subdir": "nifti_outputs",
    },
     {
        "name": "net4_bias_wbm",
        "save_dir": "../redoing_stuff_12_14/network4_bias",
        "nifti_subdir": "nifti_outputs",
    },
    {
        "name": "net3_wbm",
        "save_dir": "../redoing_stuff_12_11/checkpoints_network3_with_brainmask",
        "nifti_subdir": "nifti_outputs",
    },
    {
        "name": "net1_2echo_wbm",
        "save_dir": "../redoing_stuff_12_11/checkpoints_net1_2echoes_with_brainmask_echo1_echo4",
        "nifti_subdir": "nifti_outputs",
    },
        {
        "name": "net1_3echo_wbm",
        "save_dir": "../redoing_stuff_12_11/checkpoints_net1_3echoes_with_brainmask",
        "nifti_subdir": "nifti_outputs",
    },
            {
        "name": "net1_4echo_wbm",
        "save_dir": "../redoing_stuff_12_11/checkpoints_net1_4echoes_with_brainmask",
        "nifti_subdir": "nifti_outputs",
    },
                {
        "name": "net1_4echo_noisy_wbm",
        "save_dir": "../redoing_stuff_12_11/checkpoints_net1_4echoes_with_brainmask_noisy",
        "nifti_subdir": "nifti_outputs",
    },
     {
        "name": "net2_4echo_noisy_wbm",
        "save_dir": "../redoing_stuff_12_11/checkpoints_network2_allechoes_with_brainmask_noisy",
        "nifti_subdir": "nifti_outputs",
    },
        {
        "name": "net4_channel_wbm",
        "save_dir": "../redoing_stuff_12_11/checkpoints_network4_with_brainmask",
        "nifti_subdir": "nifti_outputs",
    },
    # add others as needed
]

# ----------------------------
# HELPERS
# ----------------------------

def build_brain_mask(subject_id, H, W, Z):
    echo1_path = os.path.join(PROCESSED_DIR, f"{subject_id}_echo1.npy")
    if not os.path.exists(echo1_path):
        raise FileNotFoundError(f"Echo1 not found for {subject_id}")

    echo1 = np.load(echo1_path)
    nonzero = echo1[echo1 > 0]
    thr = np.percentile(nonzero, MASK_PERCENTILE)
    mask = echo1 > thr

    mask = binary_closing(mask, structure=np.ones((3, 3, 3)))
    mask = binary_fill_holes(mask)

    labels = label(mask)
    largest_cc = np.argmax(np.bincount(labels.flat)[1:]) + 1
    return labels == largest_cc


# ----------------------------
# MAIN
# ----------------------------

rows = []

for exp in EXPERIMENTS:
    exp_name  = exp["name"]
    nifti_dir = os.path.join(exp["save_dir"], exp["nifti_subdir"])

    print(f"\n=== {exp_name} ===")

    pred_t1_files = sorted(glob.glob(os.path.join(nifti_dir, "*_T1rho_pred.nii.gz")))

    for t1_path in pred_t1_files:
        subject_id = os.path.basename(t1_path).replace("_T1rho_pred.nii.gz", "")

        t2_path = os.path.join(nifti_dir, f"{subject_id}_T2star_pred.nii.gz")
        if not os.path.exists(t2_path):
            continue

        pred_t1 = nib.load(t1_path).get_fdata()
        pred_t2 = nib.load(t2_path).get_fdata()

        H, W, Z = pred_t1.shape
        mask = build_brain_mask(subject_id, H, W, Z)
        mask_flat = mask.reshape(-1)

        # Handle stacked-echo predictions (e.g. net4_film)
        if pred_t1.size != mask_flat.size:
            factor = pred_t1.size // mask_flat.size
            assert pred_t1.size == factor * mask_flat.size, "Unexpected size mismatch"
            mask_flat = np.tile(mask_flat, factor)


        # ---- T1rho ----
        t1_vals = pred_t1.reshape(-1)[mask_flat]
        t1_plausible = (t1_vals >= T1RHO_MIN) & (t1_vals <= T1RHO_MAX)
        pct_t1 = 100.0 * np.sum(t1_plausible) / t1_vals.size

        # ---- T2* ----
        t2_vals = pred_t2.reshape(-1)[mask_flat]
        t2_plausible = (t2_vals >= T2_MIN) & (t2_vals <= T2_MAX)
        pct_t2 = 100.0 * np.sum(t2_plausible) / t2_vals.size

        rows.append({
            "experiment": exp_name,
            "subject": subject_id,
            "nvox_brain": int(t1_vals.size),
            "pct_T1rho_plausible": pct_t1,
            "pct_T2star_plausible": pct_t2,
        })

# ----------------------------
# SAVE + SUMMARY
# ----------------------------

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "plausible_voxel_percentages_pred.csv")
df.to_csv(csv_path, index=False)

print(f"\nSaved CSV: {csv_path}")

summary = (
    df.groupby("experiment")[["pct_T1rho_plausible", "pct_T2star_plausible"]]
    .agg(["mean", "std"])
)
print("\nExperiment-level summary (mean ± std):")
print(summary)
