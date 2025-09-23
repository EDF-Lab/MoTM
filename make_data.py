from pathlib import Path
import torch 

from src.data.data_builder import Extract_data

# DATA_PATH     = "/home/a00775/data/datasets_covar/France"
# # LIST_DATASETS = ['electricity', 'traffic', 'solar', 'ETTh1', 'ETTh2', 'weather','spanishe']
# # LIST_DATASETS = ['ETTh1_cov1']
# LIST_DATASETS = ['dat_france_2021', 'dat_france_2022', 'dat_france_2023']

# LIST_DATASETS = ['london','enedis_ldm']
# LIST_DATASETS = ['ETTh1', 'ETTh2', 'weather','spanishe']
# LIST_DATASETS = ['temperature_spain', 'pressure_spain', 'humidity_spain']

DATA_PATH     = "/mnt/gvd/projets/irma/data/bonus"
LIST_DATASETS = ['elec15T']

# DATA_PATH  = "/home/a00775/data/foundation/lotsa/split_70_10_20"
# LIST_SHORT = ['kdd2022']
# LIST_LONG  = ['bdg-2_bear', 'bdg-2_rat', 'borealis', 'covid19_energy', 'gfc12_load', 'hog', 'ideal', 'oikolab_weather', 'pdb', 'pedestrian_counts']

TRAIN_RATIO   = 0.6
TEST_RATIO    = 0.3
FLATNESS_THRE = 0.8

# DATA_PATH     = "/home/a00775/data/foundation/synthetic"
# LIST_DATASETS = ['freq1H_period1D', 'freq30T_period1W', 'freq1H_period1D1W']
# LIST_DATASETS = ['freq1H_period3H', 'freq1D_period1W']
# TRAIN_RATIO        = 0.0
# TEST_RATIO         = 1.0

CHUNK_SIZE_IN_DAYS = 7*4   # 4 weeks

SPLIT_TRAIN_IN_CHUNKS = True   # False to keep training data of shape [N_samples, T_raw] (raw time series len)
                                # True to build training data of shape [N_samples x N_chunks, chunk_size]

def export_data() -> None:

    for dataset_name in LIST_DATASETS:
        print(dataset_name)

        (Path(DATA_PATH) / dataset_name).mkdir(exist_ok=True)

        Extract = Extract_data(
            f'{DATA_PATH}/{dataset_name}/{dataset_name}.csv',
            train_ratio = TRAIN_RATIO,
            test_ratio  = TEST_RATIO,
            nb_channels = 1,
            # nsamples    = 500
        )

        if dataset_name in ['solar', 'kdd2022']: # 10T
            nb_timestamps_per_day = 24 * 6
        elif dataset_name in ['freq30T_period1W', 'dat_france_2021', 'dat_france_2022', 'dat_france_2023']: # 30T
            nb_timestamps_per_day = 24 * 2
        elif dataset_name in ['freq15T_period1D1W', 'SHMETRO', 'elec15T']: # 15T
            nb_timestamps_per_day = 24 * 4
        elif dataset_name in ['enedis_ldm', 'london', 'london_small', 'enedis_ldm_small']: # 30T
            nb_timestamps_per_day = 24 * 2
        elif dataset_name in ['largest', 'LOS_LOOP', 'PEMS_BAY', 'PEMS03']: # 5T
            nb_timestamps_per_day = 24 * 12
        elif 'cmip6' in dataset_name:
            nb_timestamps_per_day = 4 # 6H
        else: # 1H
            nb_timestamps_per_day = 24

        ##### Train data generation 

        if TRAIN_RATIO > 0.0:
            Extract.make_train_data(
                nb_timestamps_per_day, 
                nb_days_per_chunk = CHUNK_SIZE_IN_DAYS, 
                make_chunks       = SPLIT_TRAIN_IN_CHUNKS,
                flatness_threshold= FLATNESS_THRE,
                seed              = 2024
            )

            train_data = Extract.train_data

            print(f'{dataset_name} train data has shape {train_data.shape}')

            # save data
            train_data_name = 'train_data.pt' if SPLIT_TRAIN_IN_CHUNKS else 'train_data_raw.pt'
            torch.save(train_data, f'{DATA_PATH}/{dataset_name}/'+ train_data_name)
        
        # ##### Test data generation 
        
        if TEST_RATIO > 0.0:

            Extract.make_test_data(
                nb_timestamps_per_day, 
                nb_days_per_chunk = CHUNK_SIZE_IN_DAYS,
                flatness_threshold= FLATNESS_THRE,
                seed              = 2025
            )

            # Extract each test ground truth  
            ground_truth = Extract.test_data['ground_truth']

            print(f'{dataset_name} gt test data has shape {ground_truth.shape}')
            print('GT nans:',ground_truth.isnan().sum())

            # Extract each test scenario 
            data_missing_blocks_missing_1    = Extract.test_data['scenario_blocks_missing_1']
            data_missing_blocks_missing_2    = Extract.test_data['scenario_blocks_missing_2']
            data_missing_pointwise_missing_1 = Extract.test_data['scenario_pointwise_missing_1']
            data_missing_pointwise_missing_2 = Extract.test_data['scenario_pointwise_missing_2']
            # data_scenario_blocks_forecast_1  = Extract.test_data['scenario_blocks_forecast_1']
            # data_scenario_blocks_forecast_2  = Extract.test_data['scenario_blocks_forecast_2']
            # data_scenario_blocks_forecast_3  = Extract.test_data['scenario_blocks_forecast_3']
            
            torch.save(ground_truth, f'{DATA_PATH}/{dataset_name}/ground_truth.pt')
            torch.save(data_missing_blocks_missing_1, f'{DATA_PATH}/{dataset_name}/blocks_missing_1.pt')
            torch.save(data_missing_blocks_missing_2, f'{DATA_PATH}/{dataset_name}/blocks_missing_2.pt')
            torch.save(data_missing_pointwise_missing_1, f'{DATA_PATH}/{dataset_name}/pointwise_missing_1.pt')
            torch.save(data_missing_pointwise_missing_2, f'{DATA_PATH}/{dataset_name}/pointwise_missing_2.pt')
            # torch.save(data_scenario_blocks_forecast_1, f'{DATA_PATH}/{dataset_name}/blocks_forecast_1.pt')
            # torch.save(data_scenario_blocks_forecast_2, f'{DATA_PATH}/{dataset_name}/blocks_forecast_2.pt')
            # torch.save(data_scenario_blocks_forecast_3, f'{DATA_PATH}/{dataset_name}/blocks_forecast_3.pt')
            

        ##### Val data generation 

        if TRAIN_RATIO + TEST_RATIO < 1.0:

            Extract.make_val_data(
                nb_timestamps_per_day, 
                nb_days_per_chunk = CHUNK_SIZE_IN_DAYS,
                flatness_threshold= FLATNESS_THRE,
                seed              = 2026
            )

            # Extract each test ground truth  
            val_ground_truth = Extract.val_data['ground_truth']

            print(f'{dataset_name} gt val data has shape {val_ground_truth.shape}')

            # Extract each test scenario 
            val_data_missing_blocks_missing_1    = Extract.val_data['scenario_blocks_missing_1']
            val_data_missing_blocks_missing_2    = Extract.val_data['scenario_blocks_missing_2']
            val_data_missing_pointwise_missing_1 = Extract.val_data['scenario_pointwise_missing_1']
            val_data_missing_pointwise_missing_2 = Extract.val_data['scenario_pointwise_missing_2']
            # val_data_scenario_blocks_forecast_1  = Extract.val_data['scenario_blocks_forecast_1']
            # val_data_scenario_blocks_forecast_2  = Extract.val_data['scenario_blocks_forecast_2']
            # val_data_scenario_blocks_forecast_3  = Extract.val_data['scenario_blocks_forecast_3']

            # save data:
            torch.save(val_ground_truth, f'{DATA_PATH}/{dataset_name}/val_ground_truth.pt')
            torch.save(val_data_missing_blocks_missing_1, f'{DATA_PATH}/{dataset_name}/val_blocks_missing_1.pt')
            torch.save(val_data_missing_blocks_missing_2, f'{DATA_PATH}/{dataset_name}/val_blocks_missing_2.pt')
            torch.save(val_data_missing_pointwise_missing_1, f'{DATA_PATH}/{dataset_name}/val_pointwise_missing_1.pt')
            torch.save(val_data_missing_pointwise_missing_2, f'{DATA_PATH}/{dataset_name}/val_pointwise_missing_2.pt')
            # torch.save(val_data_scenario_blocks_forecast_1, f'{DATA_PATH}/{dataset_name}/val_blocks_forecast_1.pt')
            # torch.save(val_data_scenario_blocks_forecast_2, f'{DATA_PATH}/{dataset_name}/val_blocks_forecast_2.pt')
            # torch.save(val_data_scenario_blocks_forecast_3, f'{DATA_PATH}/{dataset_name}/val_blocks_forecast_3.pt')
        print()


if __name__ == '__main__':
    export_data()

