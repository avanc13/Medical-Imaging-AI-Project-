#!/usr/bin/env python3
"""
Compute percentage of LS voxels within physiologic range
inside a brain mask, over the validation set.

Outputs:
  - plausible_voxel_percentages_ls.csv
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.measure import label

# ----------------------------
# CONFIG
# ----------------------------

LS_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/ground_truth_maps"
PROCESSED_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data/processed"

OUT_DIR = "plaus_metric_outputs/param_metrics_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

MASK_PERCENTILE = 60.0

# Physiologic LS value filters (seconds)
T1RHO_MIN, T1RHO_MAX = 0.2, 3.0
T2_MIN,    T2_MAX    = 0.02, 0.18

# ----------------------------
# HELPERS
# ----------------------------

def build_brain_mask(subject_id, H, W, Z):
    echo1_path = os.path.join(PROCESSED_DIR, f"{subject_id}_echo1.npy")
    if not os.path.exists(echo1_path):
        raise FileNotFoundError(f"Echo1 not found for {subject_id}")

    echo1 = np.load(echo1_path)
    if echo1.shape != (H, W, Z):
        raise ValueError(f"Echo1 shape mismatch for {subject_id}")

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

ls_t2_files = sorted(glob.glob(os.path.join(LS_DIR, "*_T2star.npy")))
subjects = [
    os.path.basename(p).replace("_T2star.npy", "")
    for p in ls_t2_files
]

print(f"Found {len(subjects)} subjects")

rows = []

for subject_id in subjects:
    t1_path = os.path.join(LS_DIR, f"{subject_id}_T1p.npy")
    t2_path = os.path.join(LS_DIR, f"{subject_id}_T2star.npy")

    if not (os.path.exists(t1_path) and os.path.exists(t2_path)):
        continue

    ls_t1 = np.load(t1_path)
    ls_t2 = np.load(t2_path)

    H, W, Z = ls_t1.shape
    mask = build_brain_mask(subject_id, H, W, Z)
    mask_flat = mask.reshape(-1)

    # ---- T1rho ----
    t1_vals = ls_t1.reshape(-1)[mask_flat]
    t1_plausible = (t1_vals >= T1RHO_MIN) & (t1_vals <= T1RHO_MAX)

    pct_t1 = 100.0 * np.sum(t1_plausible) / t1_vals.size if t1_vals.size > 0 else np.nan

    # ---- T2* ----
    t2_vals = ls_t2.reshape(-1)[mask_flat]
    t2_plausible = (t2_vals >= T2_MIN) & (t2_vals <= T2_MAX)

    pct_t2 = 100.0 * np.sum(t2_plausible) / t2_vals.size if t2_vals.size > 0 else np.nan

    rows.append({
        "subject": subject_id,
        "nvox_brain": int(t1_vals.size),
        "pct_T1rho_plausible": pct_t1,
        "pct_T2star_plausible": pct_t2,
    })

# ----------------------------
# SAVE + SUMMARY
# ----------------------------

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "plausible_voxel_percentages_ls.csv")
df.to_csv(csv_path, index=False)

print(f"\nSaved CSV: {csv_path}")

print("\nDataset-level summary (mean ± std):")
print(df[["pct_T1rho_plausible", "pct_T2star_plausible"]].agg(["mean", "std"]))
