#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --job-name=infer
#SBATCH --output=infer.out
#SBATCH --error=infer.err

# an_all_long_rd, an_all_short

CWD=$(pwd)
SCRIPT='scripts/slurm'

if [[ "$CWD" =~ .*"$SCRIPT".* ]]; then
    cd ../../
fi

source .venv/bin/activate

# =================================================================
# User Config Section 
# =================================================================

data_dir=train
data=solar
test_setting_list=(pointwise_missing_1.pt pointwise_missing_2.pt blocks_missing_1.pt blocks_missing_2.pt)

batch_size=64                  # dataloader batch size

plot_imputation=True
nb_plots=5

model_path=outputs/saved_weights/Solar

# =================================================================
for test_setting in ${test_setting_list[@]}; do

    srun python3 -u timeflow_infer.py                             \
        "data=${data_dir}/${data}"                                \
        "data.batch_size=${batch_size}"                           \
        "data.importation.test_file=${test_setting}"              \
        "task=imputation"                                         \
        "task.plot_imputation=${plot_imputation}"                 \
        "task.nb_plots=${nb_plots}"                               \
        "++model_path=${model_path}"

done
