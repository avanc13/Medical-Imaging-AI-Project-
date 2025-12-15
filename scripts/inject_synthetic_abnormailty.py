#!/usr/bin/env python3
"""
Inject a synthetic abnormality (lesion) into LS T1rho/T2* maps
and generate corresponding abnormal echo images via the FLASH
forward model.

Outputs:
  - Abnormal LS maps (.npy) in OUT_LS_DIR:
        sub-XXXXX_T1rho_abn.npy
        sub-XXXXX_T2star_abn.npy

  - Abnormal echoes (.npy) in OUT_ECHO_DIR:
        sub-XXXXX_echo1.npy
        sub-XXXXX_echo2.npy
        sub-XXXXX_echo3.npy
        sub-XXXXX_echo4.npy

You can then:
  - Use OUT_ECHO_DIR as DATA_DIR for inference with Net1/Net2.
  - Use OUT_LS_DIR as LS_DIR in a copy of your param eval script
    to treat these abnormal maps as ground truth.
"""

import os
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.measure import label

# ----------------------------
# CONFIG
# ----------------------------

# Original LS maps
LS_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/ground_truth_maps_processed"

# Original processed echoes (for brain mask)
PROCESSED_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data/processed"

# Where to save abnormal LS maps
OUT_LS_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/ground_truth_maps_abnormal"
os.makedirs(OUT_LS_DIR, exist_ok=True)

# Where to save abnormal echoes (fake processed folder)
OUT_ECHO_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data/processed_abnormal"
os.makedirs(OUT_ECHO_DIR, exist_ok=True)

# Subject to corrupt
SUBJECT_ID = "sub-19979"  # change as needed

# Echo times (seconds) – must match training
TEs = np.array([0.012, 0.028, 0.044, 0.060], dtype=np.float32)

# Mask threshold
MASK_PERCENTILE = 60.0

# Synthetic lesion parameters (seconds)
LESION_T1RHO = 0.8   # e.g., 0.8 s
LESION_T2STAR = 0.025  # 25 ms, unusually short

# Size of lesion patch in voxels (x, y in one slice)
LESION_X0, LESION_X1 = 20, 35
LESION_Y0, LESION_Y1 = 20, 35  # adjust if needed


# ----------------------------
# HELPERS
# ----------------------------

def build_brain_mask_and_echo1(subject_id, H, W, Z):
    """3D brain mask from echo1, similar to your eval code."""
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

    return mask, echo1


def flash_forward(T1rho, T2s, TEs):
    """
    FLASH forward model in numpy:

      y(TE) = T1rho * exp(-TE / T2*)

    T1rho, T2s: (H, W, Z)
    TEs: (N_echoes,)
    returns echoes: (N_echoes, H, W, Z)
    """
    eps = 1e-3
    T2s_safe = np.abs(T2s) + eps
    T1_safe = np.abs(T1rho) + eps

    echoes = []
    for TE in TEs:
        y = T1_safe * np.exp(-TE / T2s_safe)
        echoes.append(y.astype(np.float32))

    return np.stack(echoes, axis=0)  # (N_echoes, H, W, Z)


# ----------------------------
# MAIN
# ----------------------------

def main():
    # ---- Load original LS maps ----
    t1_path = os.path.join(LS_DIR, f"{SUBJECT_ID}_T1rho_ls.npy")
    t2_path = os.path.join(LS_DIR, f"{SUBJECT_ID}_T2star_ls.npy")
    if not (os.path.exists(t1_path) and os.path.exists(t2_path)):
        raise FileNotFoundError(f"Missing LS maps for {SUBJECT_ID} in {LS_DIR}")

    T1_ls = np.load(t1_path)  # (H, W, Z)
    T2_ls = np.load(t2_path)
    if T1_ls.shape != T2_ls.shape:
        raise ValueError(f"LS T1rho and T2* shapes differ for {SUBJECT_ID}")

    H, W, Z = T1_ls.shape
    print(f"{SUBJECT_ID}: LS shape = {T1_ls.shape}")

    # ---- Build brain mask to keep lesion inside brain ----
    brain_mask, echo1 = build_brain_mask_and_echo1(SUBJECT_ID, H, W, Z)
    print(f"Brain mask voxels: {brain_mask.sum()}")

    # ---- Define lesion ROI on mid slice and AND with brain mask ----
    z0 = Z // 2
    lesion_mask = np.zeros_like(T1_ls, dtype=bool)
    lesion_mask[LESION_X0:LESION_X1, LESION_Y0:LESION_Y1, z0] = True

    # Force lesion to be inside brain
    lesion_mask = lesion_mask & brain_mask
    n_lesion_vox = lesion_mask.sum()
    if n_lesion_vox == 0:
        raise RuntimeError("Lesion mask ended up empty; adjust LESION_* or MASK_PERCENTILE.")

    print(f"Lesion voxels: {n_lesion_vox}")

    # ---- Build abnormal LS maps ----
    T1_abn = T1_ls.copy()
    T2_abn = T2_ls.copy()

    T1_abn[lesion_mask] = LESION_T1RHO
    T2_abn[lesion_mask] = LESION_T2STAR

    # Save abnormal LS maps
    out_t1_abn = os.path.join(OUT_LS_DIR, f"{SUBJECT_ID}_T1rho_abn.npy")
    out_t2_abn = os.path.join(OUT_LS_DIR, f"{SUBJECT_ID}_T2star_abn.npy")
    np.save(out_t1_abn, T1_abn.astype(np.float32))
    np.save(out_t2_abn, T2_abn.astype(np.float32))
    print(f"Saved abnormal LS maps:\n  {out_t1_abn}\n  {out_t2_abn}")

    # ---- Generate abnormal echoes from abnormal LS maps ----
    echoes_abn = flash_forward(T1_abn, T2_abn, TEs)  # (N_echoes, H, W, Z)

#no need for norm again

    for e_idx in range(echoes_abn.shape[0]):
        echo_vol = echoes_abn[e_idx]  # (H, W, Z)
        out_echo_path = os.path.join(OUT_ECHO_DIR, f"{SUBJECT_ID}_echo{e_idx+1}.npy")
        np.save(out_echo_path, echo_vol.astype(np.float32))
        print(f"Saved abnormal echo {e_idx+1}: {out_echo_path}")

    print("\nDone. You can now:")
    print(f"  - Use {OUT_ECHO_DIR} as DATA_DIR for Net1/Net2 inference.")
    print(f"  - Use {OUT_LS_DIR} as LS_DIR (abnormal) for evaluating how well each network")
    print("    recovers the injected lesion parameters inside the lesion_mask region.")


if __name__ == "__main__":
    main()
