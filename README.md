# EnsembleNGP

This is the official code for paper: EnsembleNGP: Exploring Time-Varying Volumetric Ensemble Data via Parameter and Spatial Space Decomposition

# Required Libary

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

You can modify the config file under configs to configure the model settings, such as log2_map_size in NGP and parameter emebedding size. 
