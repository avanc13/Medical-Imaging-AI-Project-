import os
import numpy as np

PROCESSED_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/data/processed"  # clean, no noise
OUT_DIR = "/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/ground_truth_maps_processed"
os.makedirs(OUT_DIR, exist_ok=True)

TE_values = [0.012, 0.028, 0.044, 0.06]
BAD_SUBJECTS = {"sub-04620"} # just hardcode this corrupted one for now


def fit_voxel_linear(signal, TE_values):
    if np.all(signal <= 0):
        return 0.0, 0.0

    log_signals = np.log(np.clip(signal, 1e-6, None)) # take log, avoid log(0)
    TE_values = np.asarray(TE_values, dtype=float) #
    X = np.column_stack([np.ones(len(TE_values)), TE_values]) # design matrix that we will compute least squares on
    A, B = np.linalg.lstsq(X, log_signals, rcond=None)[0]

    T1p = np.exp(A)
    T2star = -1.0 / B if B != 0 else 0.0 # avoid division by zero
    if B >= 0 or T2star <= 0 or T2star > 1.0:
        return 0.0, 0.0
    return T1p, T2star

# discover subjects from processed echo1
subject_ids = sorted(
    fname.replace("_echo1.npy", "")
    for fname in os.listdir(PROCESSED_DIR)
    if fname.endswith("_echo1.npy")
)

for sid in subject_ids:
    if sid in BAD_SUBJECTS:
        print(f"Skipping corrupted subject: {sid}")
        continue
    print("Processing", sid)
    # load the 4 processed echoes you used for training
    echoes = []
    for i in range(1, 5):
        path = os.path.join(PROCESSED_DIR, f"{sid}_echo{i}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        echoes.append(np.load(path))  # (H,W,Z)

    echoes = np.stack(echoes, axis=0)  # (4,H,W,Z)
    _, H, W, Z = echoes.shape

    T1p_map   = np.zeros((H, W, Z), dtype=np.float32)
    T2star_map = np.zeros((H, W, Z), dtype=np.float32)

    for x in range(H):
        for y in range(W):
            for z in range(Z):
                signal = echoes[:, x, y, z]
                T1p, T2star = fit_voxel_linear(signal, TE_values)
                T1p_map[x, y, z] = T1p
                T2star_map[x, y, z] = T2star

    np.save(os.path.join(OUT_DIR, f"{sid}_T1rho_ls.npy"), T1p_map)
    np.save(os.path.join(OUT_DIR, f"{sid}_T2star_ls.npy"), T2star_map)
