import os
import numpy as np
import nibabel as nib

def fit_voxel_linear(signal, TE_values):
    """Fit T1ρ and T2* using linear model: ln(y) = A + B * TE."""
    # Skip voxels with no usable signal
    if np.all(signal <= 0):
        return 0.0, 0.0

    # Take natural log of signals
    log_signals = np.log(np.clip(signal, 1e-6, None))

    # Design matrix: [1, TE] for each echo time
    TE_values = np.asarray(TE_values, dtype=float)
    design_matrix = np.column_stack([np.ones(len(TE_values)), TE_values])

    # Solve linear system: A + B * TE = ln(y)
    A, B = np.linalg.lstsq(design_matrix, log_signals, rcond=None)[0]

    # Convert back to parameters: A = ln(T1ρ), B = -1 / T2*
    T1p = np.exp(A)
    T2star = -1.0 / B if B != 0 else 0.0
  
    if B >= 0 or T2star <= 0 or T2star > 1.0:
       return 0.0, 0.0

    return T1p, T2star


def create_parameter_maps(data_root, subject_id, TE_values):
    """Create T1ρ and T2* maps for one subject."""
    subject_path = os.path.join(data_root, subject_id)

    # Load all 4 echo time images
    echoes = []
    for i in range(1, 5):
        file_name = f"{subject_id}_task-rest_echo-{i}_bold.nii.gz"
        file_path = os.path.join(subject_path, "func", file_name)
        img_data = nib.load(file_path).get_fdata()
        echoes.append(img_data)

    echoes = np.array(echoes) # Shape: [4, 64, 64, 30, 239]

    # Average over the time dimension
    echoes_avg = np.mean(echoes, axis=-1) # Shape: [4, 64, 64, 30]

    nx, ny, nz = echoes_avg.shape[1:]
    T1p_map = np.zeros((nx, ny, nz), dtype=np.float32)
    T2star_map = np.zeros((nx, ny, nz), dtype=np.float32)

    # Fit parameters for each voxel using the linear method
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                signal = echoes_avg[:, x, y, z]  # 4 signal values (one per echo)
                T1p, T2star = fit_voxel_linear(signal, TE_values)
                T1p_map[x, y, z] = T1p
                T2star_map[x, y, z] = T2star

    return T1p_map, T2star_map


DATA_ROOT = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data"

# saving ground truth in project root, not inside the /data path for now
PARAM_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/ground_truth_maps"

os.makedirs(PARAM_DIR, exist_ok=True)

TE_values = [0.012, 0.028, 0.044, 0.06]

# All subject folders in /data path that follow "sub-XXXXX"
subject_ids = sorted(
    d for d in os.listdir(DATA_ROOT)
    if d.startswith("sub-") and os.path.isdir(os.path.join(DATA_ROOT, d))
)

BAD_SUBJECTS = {"sub-04620"}
for subject_id in subject_ids:
    if subject_id in BAD_SUBJECTS:
        print(f"Skipping corrupted subject: {subject_id}")
        continue

    print(f"Processing {subject_id}...")
    T1p_map, T2star_map = create_parameter_maps(DATA_ROOT, subject_id, TE_values)

    np.save(os.path.join(PARAM_DIR, f"{subject_id}_T1p.npy"), T1p_map)
    np.save(os.path.join(PARAM_DIR, f"{subject_id}_T2star.npy"), T2star_map)
