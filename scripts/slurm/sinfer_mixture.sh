#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --job-name=timeflow
#SBATCH --output=infer.out
#SBATCH --error=infer.err

CWD=$(pwd)
SCRIPT='scripts/slurm'

if [[ "$CWD" =~ .*"$SCRIPT".* ]]; then
    cd ../../
fi

source .venv/bin/activate

# =================================================================
# User Config Section 
# =================================================================

subdir_name=id                       # where to export the inference results

data_dir=train
data_list=(electricity solar spanishw_t)

inr_last_layers=1                      # number of layers to consider for features

# test_setting_list=(blocks_missing_1.pt blocks_missing_2.pt)
# lambda_ridge=1                        # regularization coeff for ridge regression

test_setting_list=(pointwise_missing_1.pt pointwise_missing_2.pt)
lambda_ridge=0.5                      # regularization coeff for ridge regression

learn_lambda=False                     # whether to estimate ridge lambda

run_baselines=False                    # whether to compute Linear and Offset baselines

plot_imputation=False                  # whether to make some plots at inference
nb_plots=5                             # number of inference plots

# list all pretrained TimeFlows here:
model_path_1=outputs/saved_weights/Electricity/
model_path_2=outputs/saved_weights/Solar/
model_path_3=outputs/saved_weights/SpanishW-T/

# add them manually to a list:
list_models="[$model_path_1, $model_path_2, $model_path_3]"

# =================================================================

for data in ${data_list[@]}; do

    if [ "$data" == "solar" ]; then
        batch_size=64                          # dataloader batch size

    else
        batch_size=256                         # dataloader batch size

    fi

    for test_setting in ${test_setting_list[@]}; do

        srun python3 -u timeflow_infer_mixture.py                     \
            "hydra=mixture"                                           \
            "data=${data_dir}/${data}"                                \
            "data.batch_size=${batch_size}"                           \
            "data.importation.test_file=${test_setting}"              \
            "task=imputation"                                         \
            "task.plot_imputation=${plot_imputation}"                 \
            "task.nb_plots=${nb_plots}"                               \
            "task.run_baselines=${run_baselines}"                     \
            "++inr_last_layers=${inr_last_layers}"                    \
            "++lambda_ridge=${lambda_ridge}"                          \
            "++learn_lambda=${learn_lambda}"                          \
            "++subdir_name=${subdir_name}"                            \
            "++list_model_path=${list_models}"

    done

done

srun python3 -u src/tools/utils/read_results.py --p outputs/mixture/${subdir_name}