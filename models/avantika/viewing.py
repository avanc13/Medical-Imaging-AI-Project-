import numpy as np
# if not hasattr(np, "float64"):
#     np.float64 = float
# import nibabel as nib

# img = nib.load("/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data/sub-04570/func/sub-04570_task-rest_echo-3_bold.nii.gz")
# data = img.get_fdata()
# print(data.shape)

# import nibabel as nib
# import os, glob

# folder = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data/sub-04570/func"
# for path in sorted(glob.glob(os.path.join(folder, "*echo-*bold.nii.gz"))):
#     try:
#         img = nib.load(path)
#         data = img.get_fdata()
#         print(f"{os.path.basename(path)}  shape={data.shape}")
#     except Exception as e:
#         print(f" {os.path.basename(path)} failed: {type(e).__name__}: {e}")

print(np.load("/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data/processed/sub-04570_echo1.npy").shape   )