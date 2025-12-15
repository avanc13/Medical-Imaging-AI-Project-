#!/usr/bin/env python3
"""
preprocess multi-echo FLASH MRI data with bias-field correction.

Steps:
1) Load all echo files per subject
2) Average across timesteps (4D -> 3D)
3) Estimate and remove smooth intensity bias field
4) Normalize intensities across echoes (shared p99)
5) Save per-echo .npy volumes into data/processed_bias_corrected
"""

import os
import glob
import numpy as np
import nibabel as nib

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


RAW_ROOT = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data"
OUT_DIR = os.path.join(RAW_ROOT, "processed_bias_corrected")
LOG_PATH = os.path.join(OUT_DIR, "corrupted_files.txt")
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_average(path: str) -> np.ndarray:
    img = nib.load(path)
    data = img.get_fdata()
    if data.ndim == 4:
        data = data.mean(axis=-1)
    return data


def normalize_subject_volumes(echo_volumes):
    stacked = np.stack(echo_volumes, axis=-1)
    p99 = np.percentile(stacked, 99)

    normalized = []
    for vol in echo_volumes:
        vol_norm = np.clip(vol / (p99 + 1e-8), 0, 1).astype(np.float32)
        normalized.append(vol_norm)
    return normalized


def save_volume(volume: np.ndarray, subject_id: str, echo_index: int):
    out_path = os.path.join(OUT_DIR, f"{subject_id}_echo{echo_index}.npy")
    np.save(out_path, volume)
    print(f" Saved {out_path}  shape={volume.shape}")


def _to_sitk_image(volume_xyz: np.ndarray) -> "sitk.Image":
    volume_zyx = np.transpose(volume_xyz, (2, 1, 0))
    return sitk.GetImageFromArray(volume_zyx.astype(np.float32))


def _from_sitk_image(image: "sitk.Image") -> np.ndarray:
    volume_zyx = sitk.GetArrayFromImage(image)
    return np.transpose(volume_zyx, (2, 1, 0))


def remove_bias_field(echo_volumes, shrink_factor=4):
    if sitk is None:
        raise ImportError(
            "SimpleITK is required for bias-field correction. "
            "Install it in your environment to proceed."
        )

    # use first echo as reference
    reference = np.clip(echo_volumes[0], 0, None)
    ref_img = _to_sitk_image(reference)

    # rescale for stable thresholding
    ref_img = sitk.RescaleIntensity(ref_img, 0, 255)

    # foreground mask
    mask = sitk.OtsuThreshold(ref_img, 0, 1, 200)

    # shrink image and mask for bias estimation
    shrink = [shrink_factor] * ref_img.GetDimension()
    ref_small = sitk.Cast(sitk.Shrink(ref_img, shrink), sitk.sitkFloat32)
    mask_small = sitk.Cast(sitk.Shrink(mask, shrink), sitk.sitkUInt8)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(ref_small, mask_small)

    # get full-resolution bias field
    log_bias = corrector.GetLogBiasFieldAsImage(ref_img)
    bias_field = sitk.Exp(log_bias)

    corrected_volumes = []
    for vol in echo_volumes:
        vol_img = _to_sitk_image(np.clip(vol, 0, None))
        vol_corrected = sitk.Divide(vol_img, bias_field)
        corrected_volumes.append(_from_sitk_image(vol_corrected))

    return corrected_volumes


def main():
    subjects = sorted(glob.glob(os.path.join(RAW_ROOT, "sub-*")))
    corrupted_entries = []

    for subject_path in subjects:
        subject_id = os.path.basename(subject_path)
        echo_paths = sorted(
            glob.glob(os.path.join(subject_path, "func", "*echo-*bold.nii.gz"))
        )

        if not echo_paths:
            continue

        print(f"\nProcessing {subject_id} with {len(echo_paths)} echoes")

        echo_volumes = []
        echo_indices = []

        for idx, echo_path in enumerate(echo_paths, start=1):
            try:
                vol = load_and_average(echo_path)
                echo_volumes.append(vol)
                echo_indices.append(idx)
            except Exception as e:
                corrupted_entries.append(
                    f"{subject_id}/{os.path.basename(echo_path)}: {e}"
                )

        if not echo_volumes:
            continue

        try:
            echo_volumes = remove_bias_field(echo_volumes)
        except Exception as e:
            corrupted_entries.append(f"{subject_id}/bias_correction: {e}")
            continue

        echo_volumes = normalize_subject_volumes(echo_volumes)

        for idx, vol in zip(echo_indices, echo_volumes):
            save_volume(vol, subject_id, idx)

    if corrupted_entries:
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(corrupted_entries))


if __name__ == "__main__":
    main()

