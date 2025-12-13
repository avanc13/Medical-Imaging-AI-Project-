"""
in the preprocess.py file, if we seek to perform field bias correction,then the 
function remove_bias_field may be called right before normalization of the echo volumes
"""

import numpy as np
import nibabel as nib

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

def _to_sitk_image(volume_xyz: np.ndarray) -> "sitk.Image":
    volume_zyx = np.transpose(volume_xyz, (2, 1, 0))
    return sitk.GetImageFromArray(volume_zyx.astype(np.float32))


def _from_sitk_image(image: "sitk.Image") -> np.ndarray:
    volume_zyx = sitk.GetArrayFromImage(image)
    return np.transpose(volume_zyx, (2, 1, 0))


def remove_bias_field(echo_volumes, shrink_factor=4):
    if sitk is None:
        raise ImportError(
            "SimpleITK is required for field-bias correction in your environment. "
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