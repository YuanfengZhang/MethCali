from argparse import ArgumentParser, Namespace
from datetime import datetime
import logging
from pathlib import Path
import warnings
from autogluon.tabular import TabularPredictor
import pandas as pd
import polars as pl


warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_combination(dataset_files: list[Path], combination: str) -> tuple[list[Path],
                                                                            list[Path]]:
    if combination in ['0', '1', '2', '3', '4']:
        validate_files = [f for f in dataset_files
                          if 'train' in f.name or f'_{combination}' in f.name]
    elif len(combination) == 6 and all(c in ['0', '1', '2', '3', '4'] for c in combination):
        patterns = [f'D5_{combination[0]}', f'D6_{combination[1]}',
                    f'F7_{combination[2]}', f'M8_{combination[3]}',
                    f'BC_{combination[4]}', f'BL_{combination[5]}']
        validate_files = [f for f in dataset_files
                          if 'train' in f.name or any(p in f.name for p in patterns)]
    else:
        raise ValueError(f'Invalid combination: {combination}. '
                         'Must be a single digit from 0 to 4 or a 6-digit string.')

    training_files = [f for f in dataset_files if f not in validate_files]

    return training_files, validate_files


def main():
    arg_parser: ArgumentParser = ArgumentParser(description='AutoGluon Sampling')
    arg_parser.add_argument('--dataset-dir', type=str, required=True, help='Path to the dataset directory')
    arg_parser.add_argument('--model-dir', type=str, required=True, help='Path to the model directory')
    arg_parser.add_argument('--combination', type=str, dest='combination', default='4',
                            help='Pick a specific combination of dataset to build the model, '
                                 'can either be a single number from 0 to 4, or a 6-digit string. '
                                 'For instance, (1) 0 means all the datasets from all samples with index 0 '
                                 'will be used for validation and others will be used for training. '
                                 'Same for 1, 2, 3, and 4. (2) 012432 means the D5_0, D6_1, F7_2, M8_4, '
                                 'BC_3 AND BL_2 will be used for validation.')
    args: Namespace = arg_parser.parse_args()

    dataset_dir: Path = Path(args.dataset_dir)
    model_dir: Path = Path(args.model_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f'Dataset directory {dataset_dir} does not exist or is not a directory.')

    model_dir.parent.mkdir(parents=True, exist_ok=True)

    train_files: list[Path]
    validate_files: list[Path]

    (train_files,
     validate_files) = parse_combination(dataset_files=list(dataset_dir.glob('*parquet.lz4')),
                                         combination=args.combination)

    logger.info(f'Training files:\n{train_files}\n\nValidation files:\n{validate_files}')

    train_data: pd.DataFrame = (pl.concat([pl.scan_parquet(f) for f in train_files])
                                  .drop('chrom', 'start', 'end')
                                  .collect()
                                  .to_pandas())
    validate_data: pd.DataFrame = (pl.concat([pl.scan_parquet(f) for f in validate_files])
                                     .drop('chrom', 'start', 'end')
                                     .collect()
                                     .to_pandas())
    logger.info(f'Training data shape: {train_data.shape},\n'
                f'Validation data shape: {validate_data.shape}')

    timestamp: str = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger.info(f'start training at {timestamp}')

    # predictor = (TabularPredictor(label='actual_beta', eval_metric='root_mean_squared_error',
    #                               path=model_dir / timestamp, log_to_file=True,
    #                               log_file_path=model_dir / f'{timestamp}.log')
    #              .fit(train_data=train_data, presets='experimental_quality', fit_strategy='sequential',
    #                   num_cpus=120, num_gpus=0, ag_args_fit={'num_cpus': 120, 'num_gpus': 0},
    #                   num_bag_folds=5, excluded_model_types=['KNN'],
    #                   time_limit = 24 * 3600))
    predictor = (TabularPredictor(label='actual_beta', eval_metric='root_mean_squared_error',
                                  path=model_dir / timestamp, log_to_file=True,
                                  log_file_path=model_dir / f'{timestamp}.log')
                 .fit(train_data=train_data, presets='experimental_quality', fit_strategy='sequential',
                      num_cpus=1, num_gpus=1, ag_args_fit={'ag.num_cpus': 1, 'ag.num_gpus': 1},
                      num_bag_folds=5, excluded_model_types=['KNN'],
                      time_limit = 24 * 3600))
    logger.info('training completed')

    with open(model_dir / timestamp / 'evaluate_stats.tsv', 'w+') as f:
        for k, v in predictor.evaluate(validate_data).items():
            f.write(f'{k}\t{v}\n')

    logger.info('Generating leaderboard')
    (predictor.leaderboard(validate_data)
              .to_csv(model_dir / timestamp / 'leaderboard.csv',
                      header=True, index=False))

    logger.info('Generating feature importance')
    (predictor.feature_importance(validate_data)
              .to_csv(model_dir / timestamp / 'feature_importance.csv',
                      header=True, index=True))
    logger.info('Completed processing')


if __name__ == '__main__':
    main()
