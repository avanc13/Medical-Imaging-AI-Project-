## Rapid tissue parameter mapping via random FLASH synthesis

This project aims to estimate quantitative tissue parameters (T2* and T1​ρ) from multi-echo FLASH MRI images using a deep learning model. We train and evaluate several neural network architectures in supervised, unsupervised, and self-supervised settings. Key points:  

- Estimate **T2*** and **T1ρ** from multi-echo FLASH MRI.  
- Train and evaluate multiple **neural network architectures**: supervised, unsupervised, and self-supervised.  
- `models/` folder contains the **model architectures** (U-Net variants) for parameter-map prediction and shared building blocks.  
- `scripts/` folder contains **scripts** for preprocessing, training, and evaluation:  
  - `check_corrupted.py`: This script checks all .nii.gz files in a specified MRI dataset and reports any that are corrupted or unreadable. It prints a summary of all problematic files at the end. 
  - `compute_param_metrics_vs_ls.py`: This script compares predicted T1ρ and T2* maps to least-squares (LS) maps, computing voxel-wise metrics within a brain mask and physiologic ranges, and saves side-by-side visualizations. It outputs a CSV summarizing mean absolute errors and correlations for each subject and experiment.  
  - `compute_percent_plaus_ls.py`: This script computes the percentage of LS T1ρ and T2* voxels that fall within physiologic ranges inside a brain mask for each subject and experiment. It outputs a CSV and prints experiment-level mean and standard deviation summaries.  
  - `compute_percent_plaus_pred.py`: This script computes the percentage of predicted T1ρ and T2* voxels that fall within physiologic ranges inside a brain mask for each subject and experiment. It outputs a CSV and prints experiment-level mean and standard deviation summaries.  
  - `compute_recon_metrics.py`: This script computes brain-masked reconstruction metrics (mean absolute residual and MSE) for each subject, echo, and experiment, and saves both a CSV and per-experiment bar plots. It also generates example mid-slice images showing echo1 and the mean absolute residual map.
  - `preprocess_bias_corrected.py`: This script preprocesses multi-echo FLASH MRI by averaging 4D volumes to 3D, performing bias-field correction, normalizing intensities across echoes, and saving the processed volumes as .npy files. It also logs any corrupted files or errors encountered during processing.
  - `preprocess.py`: This script preprocesses multi-echo FLASH MRI by averaging each 4D echo to 3D, normalizing intensities, saving them as .npy files, and logging any corrupted or unreadable files.
  - `inject_synthetic_abnormality.py`: add a synethetic lesion to input MRI
  - `create_gt_maps_correct.py`: LS parameter map creation script
- `dataloader/` folder contains code to **load the data**:  
  - `proj_prelim.ipynb`: preliminary inspection of raw data  
  - `flash_dataset.py`: dataset class for model training  
- `train/` folder contains the **network implementations and training scripts**.  
  -  `network_1.py`: self-supervised UNet
  -  `network_1_redo_with_brainmask.py`: self_supervised UNet with brainmask used during training
  -  `network2_avantika`: supervised UNet
  -  `network_3.py`: unsupervised UNet
  -  `network_4.py`: self-supervised with TE as input as an additional channel to the network
  -  `network_4_film.py`: self-supervised with TE as input using FiLM, a simple addition of MLP
  -  `network_4_bias.py`: self-supervised with TE as input as a bias in the UNet
  -  Four networks are implemented:
    1. **Network 1**: self-supervised (predict parameters → synthesize echoes → reconstruction loss)  
    2. **Network 2**: supervised-to-LS (train against least-squares parameter maps)  
    3. **Network 3**: unsupervised TE-mismatched (input echoes at some TEs, synthesize echoes at other TEs)  
    4. **Network 4**: TE-conditioned (single-echo + TE map input; learns TE-aware parameter estimation)
- `container/` folder contains Docker/Singularity configuration for inference.

---

### Singularity Image

The prebuilt Singularity image is available on the BU SCC project share:
This will be for inference on network 1-3:

/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/containers/flash-mri-n123.sif

The container includes all required code and dependencies.  
**Pretrained model checkpoints are stored separately and must be specified explicitly.**

Pretrained Checkpoints:

/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/checkpoints/for_graders
├── network1.pth   # Network 1: self-supervised
├── best_network2_unet.pth   # Network 2: supervised-to-LS
└── best_network3_unet.pth   # Network 3: Unsupervised (new TE reoconstruct)

**These were all trained with normalized data, all 4 echoes (excpet 3- trained with input echo 1 and 4), left out the additional variations to not overcrowd readme**

Example usage: 

cd /projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI

singularity exec -B $PWD:/proj container/flash-mri-n123.sif \
  python /proj/inference/run_inference_n123.py \
    --net 3 \
    --input /proj/data/processed_abnormal \
    --output /proj/testing_inference_sif \
    --checkpoint /proj/redoing_stuff_12_11/checkpoints_network3_with_brainmask/best_network3_unet.pth \
    --echo_indices 1 2 \
    --device cpu

**For network 4**, below is the checkpoint for one of the implementations:
/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/checkpoints/for_graders/best_network4_unet.pth

This is a trained checkpoint for adding a TE scalar as an input with an additional channel.

Singularity Image:  /projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/flash-mri-n4.sif

Example usage of inference script:
cd /projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI

singularity exec --cleanenv \
  -B $PWD:/workspace \
  flash-mri-n4.sif \
  python /workspace/inference/run_inference_n4.py \
    --data_dir /workspace/data/processed_abnormal \
    --subject sub-19979 \
    --checkpoint /workspace/<PATH_TO_NET4_CHECKPOINT>.pth \
    --output /workspace/testing_inference_n4/sub-19979 \
    --tes 0.005 0.010 0.015 0.020 \
    --device auto
