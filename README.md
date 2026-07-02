# EnsembleNGP

This is the official code for our paper: EnsembleNGP: Exploring Time-Varying Volumetric Ensemble Data via Parameter and Spatial Space Decomposition

## Required Libraries

This code is based on [Pytorch](https://pytorch.org) and [Tiny Cuda NN](https://github.com/nvlabs/tiny-cuda-nn). 

All results we reported in the paper are based on the Pytorch 1.11 and CUDA 11.3 under Linux System.

## Dataset Preparation
To generate the datasets we evaluate in our paper, please refer to the following websites: [Nyx](https://amrex-astro.github.io/Nyx/), [Castro](https://amrex-astro.github.io/Castro/), and [MPAS Ocean](https://mpas-dev.github.io/ocean/ocean.html).

## How to use the code
Take the Nyx dataset as an example:

- To run training, use the following command:
  ```bash
  python main.py --config_file Nyx.yaml --mode train --device 0
  ```

- To run inference, use the following command:
- ```bash
  python main.py --config_file Nyx.yaml --mode inf --device 0
  ```

You can modify the .yaml file under configs folder to configure the model settings, such as log2_map_size in NGP and parameter emebedding size. 
