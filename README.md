# Mixture of TimeFlow Models (MoTM), a Foundation Model for Time Series Imputation  

## 1. Overview  

This repository provides
 1. An implementation of **TimeFlow**, a deep learning model for time series, jointly developed by **EDF R&D** and **Sorbonne Université**;
 2. An implementation of **MoTM**, a step towards a foundation model for time series imputation. 

In a nutshell, TimeFlow is a **time-continuous deep learning model** designed for time series **imputation** and **forecasting** (this repository focuses on the imputation framework). It leverages **implicit neural representations (INRs)**, **auto-decoding**, and **meta-learning** to model and infer missing point in time series datasets. TimeFlow addresses common real-world challenges such as **irregular sampling**, **missing data**, and **unaligned multi-sensor measurements**.

**MoTM** combines a basis of INRs, each trained independently on a distinct family of time series, with a ridge regressor that adapts to the observed context
at inference.

For further details, please check:  
📖 [TimeFlow Concepts](00-concepts.md)  
📄 [TimeFlow Paper](https://arxiv.org/pdf/2306.05880)  
📄 [MoTM Paper](https://arxiv.org/pdf/2507.13207)  

### Authors  

- **Code Contributors:** Etienne Le Naour, Tahar Nabil
- **MoTM Authors:** Etienne Le Naour*, Tahar Nabil*, Ghislain Agoua
- **TimeFlow Authors:** Etienne Le Naour, Louis Serrano, Léon Migus, Yuan Yin, Ghislain Agoua, Nicolas Baskiotis, Patrick Gallinari, Vincent Guigue.  

---

## 2. Project Structure  
```
.
├── 00-concepts.md              # Notes on core concepts and background
├── README.md                   # Project documentation
├── pyproject.toml              # Project dependencies and build configuration
├── make_data.py                # Script to preprocess and generate datasets
├── timeflow_train.py           # Main training entrypoint
├── timeflow_infer.py           # Inference script (single model)
├── timeflow_infer_mixture.py   # Inference script (mixture of models)

├── config/                     # Configuration files (Hydra-compatible)
│ ├── callbacks/                # Callback settings (e.g., logging, checkpoints)
│ ├── data/                     # Dataset configurations
│ │ ├── train/                  # Training dataset YAMLs
│ │ └── infer/                  # Inference dataset YAMLs
│ ├── hydra/                    # Hydra overrides and custom settings
│ ├── inr/                      # INR (Implicit Neural Representation) configs
│ ├── optim/                    # Optimizer settings
│ ├── task/                     # Task-specific configs (e.g., imputation)
│ ├── trainer/                  # Training setup (GPU, etc.)
│ ├── imputation.yaml           # General imputation task config
│ ├── inference.yaml            # General inference task config
│ └── ...                       # Other experiment configs

├── imgs/                       # Figures used for documentation
│ ├── algo-training.png
│ ├── algo-inference.png
│ ├── INR_network.png
│ └── ...

├── outputs/                    # Experiment results
│ ├── mixture/                  # CSV logs for mixture inference
│ └── saved_weights/            # Model checkpoints per dataset
│ ├── Electricity/
│ ├── Solar/
│ └── SpanishW-T/

├── scripts/                    # Job submission scripts
│ └── slurm/                    # SLURM cluster launchers
│ ├── strain.sh                 # Training job script
│ ├── sinfer.sh                 # Inference job script
│ └── sinfer_mixture.sh         # Mixture inference job script

├── src/                        # Source code
│ ├── baselines/                # Baseline models (e.g., naive approach)
│ ├── data/                     # Data processing and loaders
│ │ ├── dataloader/             # Base and joint dataloaders
│ │ ├── interpolate.py          # Data interpolation utilities
│ │ ├── scaler.py               # Data scaling functions
│ │ └── utils.py                # Data-related helpers
│ ├── metalearning/             # Meta-learning losses and algorithms
│ ├── modules/                  # Model components
│ │ ├── freq_embedding/         # Frequency-based embeddings
│ │ ├── hypernetwork/           # Hypernetwork implementations
│ │ ├── inr/                    # INR (Implicit Neural Representations)
│ │ ├── modulation/             # Modulation layers (e.g., FiLM conditioning)
│ │ ├── ridge/                  # Ridge regression modules
│ │ └── utils.py                # Model-related utilities
│ └── tools/                    # Training and inference tools
│ ├── utils/                    # Helper utilities (plotting, masks, schedulers)
│ ├── trainer.py                # Training loop implementation
│ ├── inference.py              # Inference routines
│ └── inference_mixture.py      # Mixture inference routines
```
---

## 3. Running the Project  

### Environment Setup  

All dependencies are listed in [`pyproject.toml`](pyproject.toml).  
The scripts in [`scripts/`](scripts/) are designed for `uv`, but can be adapted for other environments like `conda`.  

### Instructions  

To run the code on a SLURM cluster, job scripts are provided in the `scripts/slurm/` directory.  
Each `.sh` file can be launched using:  

```bash
sbatch your_script.sh
```

Before launching, you should:

Modify the SLURM job script (.sh) to match your cluster environment (e.g., number of GPUs, memory, job name).

Create or edit a dataset configuration file (.yaml) inside `config/data/{train|infer}/.`

Example: `config/data/train/your_dataset.yaml`

This file defines how your dataset will be loaded and processed.

Once both are configured, you can submit the job to train, evaluate, or run inference depending on the script.

### Key Information  


- **Training**: Use `strain.sh` to launch training jobs.  
- **Inference**: Use `sinfer.sh` for single-model inference.  
- **Mixture Inference**: Use `sinfer_mixture.sh` for running mixtures of models.  
- **Dataset Configuration**: Every experiment requires a corresponding dataset `.yaml` file in `config/data/`.  
- **Outputs**: Results (checkpoints, metrics, logs) are stored in the `outputs/` folder.  

### Datasets

All data used in the MoTM work are publicly available to ensure reproducibility of the results.  
- Datasets: [zenodo datasets links](https://zenodo.org/records/17177008)

## 4. Contact  

💬 Have questions, found a bug, or need specific features?  
Feel free to reach out!  

📧 **Contact:**  
- [Etienne Le Naour](mailto:etienne.le-naour@edf.fr?subject=TimeFlow%20Imputation)  
- [Tahar Nabil](mailto:tahar.nabil@edf.fr?subject=TimeFlow%20Imputation)  

If you find this repository useful, **please consider citing TimeFlow / MoTM and starring the repo**! ⭐  
