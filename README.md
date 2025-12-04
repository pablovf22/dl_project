# GNN and semi-supervised learning for molecular property prediction - Group 98


## Project Overview

This repository contains a modular framework for training **Graph Neural Networks (GNNs)** on the **QM9 molecular dataset**, supporting both **supervised** and **semi-supervised** learning. The primary objective is to evaluate how leveraging unlabeled data can improve predictive performance when only a small subset of labeled molecules is available.

We begin by training a fully supervised GCN baseline, followed by the implementation of multiple GNN architectures, hyperparameter optimization, and two semi-supervised algorithms: **Cross Pseudo-Supervision (CPS)** and **Mean Teacher (MT)**.  

The codebase supports seamless switching between learning strategies and architectures, allowing researchers and students to reproduce experiments, compare models, and extend the framework easily.

Experiment management is handled with **Hydra** and all metrics are logged to **Weights & Biases**.


## Dataset Description

The project is based on the QM9 molecular dataset, a widely used benchmark consisting of approximately 133,000 small organic molecules. Each molecule is represented as a graph where nodes correspond to atoms and edges correspond to chemical bonds. The dataset provides 19 different continuous regression targets, including dipole moments, enthalpies, energies, heat capacities, and other quantum-chemical properties computed using Density Functional Theory (DFT). In this project, we focus on target index 2, which corresponds to one of the physical properties provided by QM9.

The data module (QM9DataModule) automatically downloads, preprocesses, shuffles, and splits the dataset according to a semi-supervised learning setup. Specifically, 72% of the dataset is used as the full training set, of which 10% becomes labeled data for supervised learning, while 90% is treated as unlabeled for semi-supervised consistency training. The remaining data are split into validation (8%) and test (10%) partitions. This ensures that hyperparameters are tuned using validation only, while the final test MSE provides an unbiased estimate of model performance.

## Installation

To run this project, you need to install the required Python packages. You can install them using pip:

```bash
# It is recommended to install PyTorch first, following the official instructions
# for your specific hardware (CPU or GPU with a specific CUDA version).
# See: https://pytorch.org/get-started/locally/

# For example, for a recent CUDA version:
# pip install torch torchvision torchaudio

# Or for CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# After installing PyTorch, install PyTorch Geometric.
# The exact command depends on your PyTorch and CUDA versions.
# Please refer to the PyTorch Geometric installation guide:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# Example for PyTorch 2.7 and CUDA 11.8
# pip install torch_geometric

# Then, install the other required packages:
pip install hydra-core omegaconf wandb pytorch-lightning numpy tqdm
```

## How to Run

The main entry point for this project is `src/run.py`. It uses `hydra` for configuration management.
To run the code, execute the following command from the root of the project:

```bash
python src/run.py
```

You can override the default configuration by passing arguments from the command line. For example, to use a different model configuration:

```bash
python src/run.py model=GCN
```

The configuration files are located in the `configs/` directory.


### Select GCN model architecture

The different GCN architectures are implemented as classes inside models.py.

You can select each architecture using the commands below, or by modifying the run.yaml config file and specifying the model you want to use.

```bash
python src/run.py model=GCN
```

```bash
python src/run.py model=GCN_v2
```

```bash
python src/run.py model=GraphSAGE
```

```bash
python src/run.py model=GIN
```

```bash
python src/run.py model=GCN_v3
```


### Select training algorithm

The different training algorithms (supervised and semi-supervised) are implemented as classes inside trainer.py.

You can select each training strategy using the commands below, or by modifying the run.yaml and specifying the method you want to use.

```bash
python src/run.py trainer=supervised-ensemble
```

```bash
python src/run.py trainer=cps-semi-supervised-ensemble
```

```bash
python src/run.py trainer=mean-teacher-semi-supervised-ensemble
```

### Weights & Biases configuration

To visualize training metrics and experiment results in **Weights & Biases (W&B)**, you must specify your personal W&B account in the logger configuration file.

Set your user account in the `entity` field:

```yaml
entity: your_wandb_username
project_name: gnn_intro
```
