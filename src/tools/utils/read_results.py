import argparse
from typing import Sequence, Optional, Union
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

def make_table(
    output_path: Path,
    list_imputation_dataset: Optional[Union[str, Sequence[str]]] = None,
    list_imputation_settings: Optional[Sequence[str]] = None,
    mae_filename: str = 'inference/metrics/mae_mixture.csv',
    metric_name: str = 'MAE',
    print_tex_table: bool = True
) -> None:
    """
    Scan `output_path` to read the csv files from `inference_mixture.py` and export to an aggregated file.

    Args:
        output_path (Path): main directory containing the experiments
        list_imputation_dataset (Sequence[str]): list of imputation datasets (= subdirs of `output_path`)
        list_imputation_settings (Sequence[str]): list of imputation settings (= subdirs of `output_path/dataset`)
        mae_filename (str): where to find the csv files
    
    Returns:
        Export an aggregated `mae_inference`.csv` at `output_path/`
    """

    if list_imputation_dataset is None:
        list_imputation_dataset = [
            '', 'Electricity', 'London', 'Solar', 'SpanishW-T',\
                'Traffic', 'ETTh1', 'ETTh2', 'Weather', 'SpanishE',
        ]
    if list_imputation_settings is None:
        list_imputation_settings = (
            'pointwise_missing_1', 'pointwise_missing_2',\
                'blocks_missing_1', 'blocks_missing_2'
        )

    results = defaultdict(list)
    list_dataset = []
    list_settting = []

    for dataset in list_imputation_dataset:

        for setting in list_imputation_settings:

            filename = output_path / dataset / setting / mae_filename
            # print(filename)

            if filename.exists():
                list_dataset.append(dataset)
                list_settting.append(setting)
                df = pd.read_csv(filename, index_col = 0)
                results['nb_chunks'].append(df['chunks'].iloc[0] if 'chunks' in df.columns else -1)
                for key in df.index:
                    results[key].append(df.loc[key]['{} on Missing Values (norm)'.format(metric_name)])
                
    if len(list_dataset) == 0:
        return
    
    df = pd.concat([pd.DataFrame({'Dataset': list_dataset, 'Setting': list_settting}), pd.DataFrame(results)], axis=1)

    mean_score = ['mean score', None, df.nb_chunks.sum()] + [
        df.iloc[:,i+3].mean() for i in range(len(df.columns)-3)
    ]
    
    if 'Ridge' in df.columns:
        improve = ['improvement (%)', None, None] + [
            100 * np.nanmean( (df.iloc[:,i+3].values - df.loc[:,'Ridge'].values)/df.iloc[:,i+3].values ) for i in range(len(df.columns)-3)
        ]
        df.loc[len(df)] = mean_score
        df.loc[len(df)] = improve
    
    # col_order = [0, 1, 5, 6, 7, 2, 3, 4]
    # df = df.iloc[:, col_order]
    # export csv:
    df.to_csv( output_path / '{}_inference.csv'.format(metric_name.lower()) )

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path')
    parser.add_argument('-d', '--datasets', default=None)
    parser.add_argument('-s', '--settings', default=None)
    parser.add_argument('-f', '--filename', default='inference/metrics/mae_mixture.csv')
    parser.add_argument('-m', '--metric', default='MAE')


    args = parser.parse_args()
    output_path = Path( args.path )


    make_table( output_path, args.datasets, args.settings, args.filename, args.metric )

