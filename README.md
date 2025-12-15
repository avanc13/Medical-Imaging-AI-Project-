## Rapid tissue parameter mapping via random FLASH synthesis
This project aims to estimate the quantitative tissue parameters (T2* and T1​ρ) from multi-echo FLASH MRI images using a deep learning model. We aim to train and evaluate several neural network architectures in supervised, unsupervised, and self-supervised settings. 

1. The `models`:Model architectures (U-Net variants) used across experiments. This includes the networks used for parameter-map prediction (T2*, T1ρ) and any shared building blocks.
2. The `scripts` folder contains scripts for preprocessing data, and training all the networks discussed in our report.
Network 1: self-supervised (predict params → synthesize echoes → reconstruction loss)

Network 2: supervised-to-LS (train against least-squares parameter maps)

Network 3: unsupervised TE-mismatched (input echoes at some TEs, train to synthesize echoes at other TEs)

Network 4: TE-conditioned (single-echo + TE map as input; learns TE-aware parameter estimation)

3. The `dataloader` folder contains the code to load the data following the file structure of the training data.
4. The `train` folder has the network implementations and training. As of now, we have Network 1 implemented.



- **`container/`**  
  Dockerfile and container configuration for inference.

---

### Singularity Image

The prebuilt Singularity image is available on the BU SCC project share:
This will be for inference on network 1-3:

/projectnb/ec500kb/projects/Fall_2025_Projects/Proj_FLASH_MRI/containers/flash-mri-n123.sif

