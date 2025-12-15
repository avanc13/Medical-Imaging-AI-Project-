## Rapid tissue parameter mapping via random FLASH synthesis
This project aims to estimate the quantitative tissue parameters (T2* and T1​ρ) from multi-echo FLASH MRI images using a deep learning model. We aim to train and evaluate several neural network architectures in supervised, unsupervised, and self-supervised settings. 

1. The `models` folder contains the model architectures (U-Net variants) used across experiments. This includes the networks used for parameter-map prediction (T2*, T1ρ) and any shared building blocks.
2. The `scripts` folder contains scripts for preprocessing data, and training all the networks discussed in our report.
  a. `check_corrupted.py`:
  b. `compute_param_metrics_vs_ls.py`:
  c. `compute_percent_plaus_ls.py`:
  d. `compute_percent_plaus_pred.py`:
  e. `compute_recon_metrics.py`:
4. The `dataloader` folder contains the code to load the data following the file structure of the training data.
  a. `proj_prelim.ipynb`: preliminary inspection of all the raw data.
  b. `flash_dataset.py`: class to load the dataset for model training.
5. The `train` folder has the network implementations and training. The network numbers and architecture are detailed below.

We have 4 networks: 
Network 1: self-supervised (predict params → synthesize echoes → reconstruction loss)

Network 2: supervised-to-LS (train against least-squares parameter maps)

Network 3: unsupervised TE-mismatched (input echoes at some TEs, train to synthesize echoes at other TEs)

Network 4: TE-conditioned (single-echo + TE map as input; learns TE-aware parameter estimation)

- **`container/`**  
  Dockerfile and container configuration for inference.

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
