#!/usr/bin/env python3
"""
checking all .nii.gz files in the MRI dataset and report corrupted or unreadable files.

"""

import os
import glob
import nibabel as nib

DATA_ROOT = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data"

def check_nifti_file(path):
    """try loading a NIfTI file. true if ok false or error message if corrupted."""
    try:
        img = nib.load(path)
        _ = img.get_fdata()  # force full read
        return True
    except Exception as e:
        return f"{type(e).__name__}: {e}"

def main():
    subjects = sorted(glob.glob(os.path.join(DATA_ROOT, "sub-*")))
    if not subjects:
        print("No subjects found in", DATA_ROOT)
        return

    corrupted = []

    for subj_path in subjects:
        subj_id = os.path.basename(subj_path)
        echo_files = sorted(glob.glob(os.path.join(subj_path, "func", "*echo-*bold.nii.gz")))

        if not echo_files:
            print(f" {subj_id}: no echo files found.")
            continue

        print(f"\n Checking {subj_id} ({len(echo_files)} echoes)")

        for echo_path in echo_files:
            fname = os.path.basename(echo_path)
            result = check_nifti_file(echo_path)
            if result is True:
                print(f"  {fname} — OK")
            else:
                print(f"  {fname} — {result}")
                corrupted.append((subj_id, fname, result))

    print("\n==================== SUMMARY ====================")
    if not corrupted:
        print(" All .nii.gz files loaded successfully")
    else:
        print(f"Found {len(corrupted)} problematic files:\n")
        for subj_id, fname, err in corrupted:
            print(f"  {subj_id}/{fname} — {err}")

if __name__ == "__main__":
    main()
