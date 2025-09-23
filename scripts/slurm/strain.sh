#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --job-name=timeflow
#SBATCH --output=train.out
#SBATCH --error=train.err

CWD=$(pwd)
SCRIPT='scripts/slurm'

if [[ "$CWD" =~ .*"$SCRIPT".* ]]; then
    cd ../../
fi

source .venv/bin/activate

# =================================================================
# User Config Section 
# =================================================================

expe_name=pretrain           # Name of the subdirectory where the experiment results will be saved

data_dir=train               # Subdirectory of config/Data here to find the dataset yaml
dataset=solar                # Name of the dataset configuration file in config/data/${data_dir}
data_weight=1.0              # Data under (<1) or over (>1) sampling strategy
batch_size=256               # Dataloader batch size

epochs=100_000               # Total number of training epochs
warmup_steps=200             # Number of warmup steps for learning rate scheduler

latent_dim=128               # Latent code dimension
hn_depth=1                   # depth of the hypernetwork
hn_width=256                 # width of the hypernetwork (if depth>1)
depth=5                      # INR depth
hidden_dim=256               # INR hidden dimension

normalize_z=False            # make sure the L2 norm of latent code is constant, or not

use_target_in_train=True     # Whether to use target data during training
lambda_target=0.5            # >=0, outer step loss is `loss_context + lambda_target x loss_target`
mask_during_training=False   # Whether to mask standard deviation samples during training

loss_type='huber'            # 'mse' or 'huber' loss function

lr_inr=1e-3                  # Learning rate for the INR network
inner_steps=3                # Number of steps in the auto-decoding process
lr_code=0.01                 # Learning rate for auto-decoding

plot_every_n_epochs=500      # Frequency of loss function visualization during training

# =================================================================

srun python3 -u timeflow_train.py                             \
    "task=imputation"                                         \
    "data=${data_dir}/${dataset}"                             \
    "data.batch_size=${batch_size}"                           \
    "data.weight=${data_weight}"                              \
    "inr.latent_dim=${latent_dim}"                            \
    "inr.hn_width=${hn_width}"                                \
    "inr.hn_depth=${hn_depth}"                                \
    "inr.width=${hidden_dim}"                                 \
    "inr.depth=${depth}"                                      \
    "inr.apply_znorm=${normalize_z}"                          \
    "trainer.name=${expe_name}"                               \
    "trainer.max_epochs=${epochs}"                            \
    "optim.use_target_for_training=${use_target_in_train}"    \
    "optim.apply_mask_during_training=${mask_during_training}"\
    "optim.lambda_target=${lambda_target}"                    \
    "optim.loss_type=${loss_type}"                            \
    "optim.lr_inr=${lr_inr}"                                  \
    "optim.lr_code=${lr_code}"                                \
    "optim.inner_steps=${inner_steps}"                        \
    "optim.scheduler.T_max=${epochs}"                         \
    "optim.scheduler.num_warmup_steps=${warmup_steps}"        \
    "callbacks.plot_freq=${plot_every_n_epochs}"
