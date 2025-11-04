#!/usr/bin/env python3
"""
preprocess multi-echo FLASH MRI data.

Steps:
1. Load all echo files per subject.
2. Average across timesteps
3. Normalize intensities
4. Save per-echo .npy volumes
5. Skip and log any corrupted or unreadable NIfTI files.--there should only be one that was bad
"""

import os
import glob
import numpy as np
import nibabel as nib

# ----------------------------
# CONFIGURATION
# ----------------------------
RAW_ROOT = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data"
OUT_DIR = os.path.join(RAW_ROOT, "processed")
LOG_PATH = os.path.join(OUT_DIR, "corrupted_files.txt")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def load_and_average(path: str) -> np.ndarray:
    img = nib.load(path)
    data = img.get_fdata()
    if data.ndim == 4:
        data = data.mean(axis=-1)
    return data

def normalize(vol: np.ndarray) -> np.ndarray:
    #scale to [0, 1] using 99th percentile
    p99 = np.percentile(vol, 99)
    vol = np.clip(vol / (p99 + 1e-8), 0, 1)
    return vol.astype(np.float32)

def save_volume(vol: np.ndarray, subj_id: str, echo_idx: int):
    # save as .npy
    out_path = os.path.join(OUT_DIR, f"{subj_id}_echo{echo_idx}.npy")
    np.save(out_path, vol)
    print(f" Saved {out_path}  shape={vol.shape}")

# ----------------------------
# MAIN PREPROCESSING LOOP
# ----------------------------
def main():
    subjects = sorted(glob.glob(os.path.join(RAW_ROOT, "sub-*")))
    if not subjects:
        print("No subjects found under", RAW_ROOT)
        return

    corrupted_files = []

    for subj_path in subjects:
        subj_id = os.path.basename(subj_path)
        echo_paths = sorted(glob.glob(os.path.join(subj_path, "func", "*echo-*bold.nii.gz")))
        if len(echo_paths) == 0:
            print(f"skipping {subj_id}: no echo files found.")
            continue

        print(f"\nProcessing {subj_id} with {len(echo_paths)} echoes...")

        for i, echo_path in enumerate(echo_paths, start=1):
            try:
                vol = load_and_average(echo_path)
                vol = normalize(vol)
                save_volume(vol, subj_id, i)

            except Exception as e:
                err_msg = f"{subj_id}/{os.path.basename(echo_path)} — {type(e).__name__}: {e}"
                print(f" Skipping corrupted echo: {err_msg}")
                corrupted_files.append(err_msg)
                continue

    # savea a log of corrupted files
    if corrupted_files:
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(corrupted_files))
        print(f"\n{len(corrupted_files)} corrupted files skipped. Logged to {LOG_PATH}")
    else:
        print("all files processed successfully — no corrupt files found.")

if __name__ == "__main__":
    main()
