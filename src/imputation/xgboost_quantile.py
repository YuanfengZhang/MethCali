from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from datetime import datetime
import gc
import logging
from pathlib import Path
from typing import Literal
import warnings
import numpy as np
import pandas as pd
from pickle import dump as pickle_dump
import polars as pl
import xgboost as xgb
import optuna
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from sklearn.model_selection import train_test_split  # type: ignore


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

SLICE_PARAMS: dict[str, int] = {
    'b5': 0, 'b4': 1, 'b3': 2, 'b2': 3, 'b1': 4,
    'a1': 6, 'a2': 7, 'a3': 8, 'a4': 9, 'a5': 10}

NUM_FEATURES: list[str] = ['predicted_beta', 'depth',
                           'GC_skew_70', 'CpG_GC_ratio_70',
                           'ShannonEntropy_70', 'BWT_ratio_70']
BOOL_FEATURES: list[str] = ['promoter', 'enhancer']
CAT_FEATURES: list[str] = ['location', 'cpg',
                           'b5', 'b4', 'b3', 'b2', 'b1',
                           'a1', 'a2', 'a3', 'a4', 'a5']
all_features: list[str] = (NUM_FEATURES + BOOL_FEATURES + CAT_FEATURES)


def read_df(f_path: Path, split: bool,
            train_test: Literal['train', 'test'],
            n_rows: int = 5000) -> pd.DataFrame:
    selected_columns: list[str] = ['predicted_beta', 'depth',
                                   'GC_skew_70', 'CpG_GC_ratio_70',
                                   'ShannonEntropy_70', 'BWT_ratio_70',
                                   'cpg', 'location', 'promoter', 'enhancer']
    df: pd.DataFrame
    if train_test == 'train':
        _df_ls: list[pl.DataFrame] = []
        for _df in (pl.scan_parquet(f_path)
                      .with_columns(pl.col('predicted_beta')
                                      .cut(breaks=list(range(5, 100, 5)),
                                           labels=[f'{i}-{i + 5}' for i in range(0, 100, 5)])
                                      .alias('beta_bin'))
                      .collect()
                      .partition_by('beta_bin', include_key=False)):
            if _df.shape[0] < n_rows:
                _df_ls.append(_df)
            else:
                _df_ls.append(_df.sample(n=n_rows))
        if split:
            selected_columns += [*SLICE_PARAMS.keys(), 'actual_beta']
            df = (pl.concat(_df_ls)
                    .lazy()
                    .with_columns(pl.when(pl.col('seq_5')
                                            .str
                                            .slice(5, 1) == 'C')
                                    .then(pl.col('seq_5'))
                                    .otherwise(pl.col('seq_5')
                                                 .str.reverse()
                                                 .str.replace_many(['A', 'C', 'G', 'T'],
                                                                   ['T', 'G', 'C', 'A']))
                                    .alias('processed_seq_5'))
                    .with_columns([pl.col('processed_seq_5')
                                     .str
                                     .slice(index, 1)
                                     .alias(name) for name, index in SLICE_PARAMS.items()])
                    .select(selected_columns)
                    .collect()
                    .to_pandas())
        else:
            selected_columns += ['seq_5', 'actual_beta']
            df = (pl.concat(_df_ls)
                    .select(selected_columns)
                    .to_pandas())
        del _df_ls
        gc.collect()
    else:
        if split:
            selected_columns += [*SLICE_PARAMS.keys()]
            df = (pl.scan_parquet(f_path)
                    .with_columns(pl.when(pl.col('seq_5')
                                            .str
                                            .slice(5, 1) == 'C')
                                    .then(pl.col('seq_5'))
                                    .otherwise(pl.col('seq_5')
                                                 .str.reverse()
                                                 .str.replace_many(['A', 'C', 'G', 'T'],
                                                                   ['T', 'G', 'C', 'A']))
                                    .alias('processed_seq_5'))
                    .with_columns([pl.col('processed_seq_5')
                                     .str
                                     .slice(index, 1)
                                     .alias(name) for name, index in SLICE_PARAMS.items()])
                    .select(['chrom', 'start', 'end'] + selected_columns)
                    .collect()
                    .to_pandas())
        else:
            selected_columns += ['seq_5']
            df = pd.read_parquet(f_path)[['chrom', 'start', 'end'] + selected_columns]
    return df


def str_cat(df: pd.DataFrame,
            columns: Iterable[str],
            choice: Literal['cat2str', 'str2cat']) -> pd.DataFrame:
    if choice == 'cat2str':
        to_type = 'category'
    else:
        to_type = 'str'

    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(to_type)

    return df


def slice_dataframe(df: pd.DataFrame, slice_size: int = 100_000) -> list[pd.DataFrame]:
    num_chunks: int = int(np.ceil(len(df) / slice_size))
    return [df.iloc[i * slice_size: (i + 1) * slice_size] for i in range(num_chunks)]


def get_xgb_params(trial: optuna.Trial,
                   gpus: int) -> dict[str, int | float | str | list[float]]:
    """Get XGBoost parameters"""
    return {
        'objective': 'reg:quantileerror',
        'quantile_alpha': [0.5],  # only train median
        'eval_metric': 'quantile',
        'tree_method': 'gpu_hist' if gpus > 0 else 'hist',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbosity': 0
    }


def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series, cpus: int, gpus: int) -> float:
    """Optuna objective function to optimize the quantile regresson"""
    params = get_xgb_params(trial, gpus)

    # Create DMatrix
    dtrain = xgb.QuantileDMatrix(X_train, y_train, nthread=cpus,
                                 enable_categorical=True)
    dval = xgb.QuantileDMatrix(X_val, y_val, ref=dtrain, nthread=cpus,
                               enable_categorical=True)

    # Train
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=[(dtrain, 'Train'), (dval, 'Test')],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # Validate
    val_pred = model.predict(dval)

    q = 0.5
    pinball_loss = np.mean(np.maximum(q * (y_val - val_pred), (q - 1) * (y_val - val_pred)))

    return float(pinball_loss)


def train_xgb_model(X_train: pd.DataFrame, y_train: pd.Series,
                    n_trials: int, timeout: int,
                    output_dir: str, cpus: int, gpus: int,
                    random_state: int = 28) -> xgb.Booster:
    """Optuna main function"""
    # split train/val datasets
    X_train_split: pd.DataFrame
    X_val: pd.DataFrame
    y_train_split: pd.Series
    y_val: pd.Series

    (X_train_split, X_val,  # type: ignore
     y_train_split, y_val) = train_test_split(  # type: ignore
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    # create Optuna study
    if Path(f'{output_dir}/optuna.sqlite3').exists():
        Path(f'{output_dir}/optuna.sqlite3').unlink()

    study: optuna.Study = optuna.create_study(direction='minimize',
                                              sampler=optuna.samplers.TPESampler(seed=random_state),
                                              pruner=optuna.pruners.HyperbandPruner(),
                                              storage=f'sqlite:///{output_dir}/optuna.sqlite3',
                                              study_name='xgboost_quantile')

    # Optimize
    study.optimize(
        lambda trial: objective(trial,
                                X_train_split, y_train_split, X_val, y_val,  # type: ignore
                                cpus, gpus),
        n_trials=n_trials,
        timeout=timeout
    )

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best pinball loss: {study.best_value:.5f}")
    logger.info(f"Best params: {study.best_params}")

    # Train final model with optimized params
    best_params = get_xgb_params(study.best_trial, gpus=gpus)  # type: ignore

    # remove n_estimators, use early stopping
    n_estimators: int = best_params.pop('n_estimators')  # type: ignore

    dtrain = xgb.QuantileDMatrix(X_train, y_train, nthread=cpus,
                                 enable_categorical=True)
    dval = xgb.QuantileDMatrix(X_val, y_val, ref=dtrain, nthread=cpus,
                               enable_categorical=True)

    model: xgb.Booster = xgb.train(
        best_params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=10
    )

    return model


def main():
    arg_parser: ArgumentParser = ArgumentParser(description=('Train XGBoost model with Optuna optimization '
                                                             'to correct low-depth methylation data.'))
    arg_parser.add_argument('-hi', '--hi-input', type=Path, required=True,
                            help='Calibrated high-depth parquet lz4 file.')
    arg_parser.add_argument('-mi', '--mi-input', type=Path, required=True,
                            help='Low-depth parquet lz4 file to be corrected now.')
    arg_parser.add_argument('-o', '--output-dir', type=Path, required=True,
                            help='Output directory to save the corrected low-depth parquet file.')
    arg_parser.add_argument('-n', '--trials', type=int, default=100,
                            help='Number of Optuna trials for hyperparameter optimization.')
    arg_parser.add_argument('--timeout', type=int, default=1800,
                            help='Timeout for hyperparameter optimization in seconds.')
    arg_parser.add_argument('-c', '--cpus', type=int, default=1,
                            help='Number of CPUs to use.')
    arg_parser.add_argument('-g', '--gpus', type=int, default=1,
                            help='Number of GPUs to use. If > 0, XGBoost will be trained on GPU.')

    args: Namespace = arg_parser.parse_args()
    hi_input_path: Path = args.hi_input
    li_input_path: Path = args.li_input
    output_dir: Path = args.output_dir
    n_trials: int = args.trials
    timeout: int = args.timeout
    cpus: int = args.cpus
    gpus: int = args.gpus

    if (output_dir / li_input_path.name).exists():
        logger.info(f'Output file {li_input_path.name} already exists in {output_dir}.')
        return

    # check input files
    for _p, _n in {hi_input_path: 'High-depth input file',
                   li_input_path: 'Low-depth input file'}.items():
        if not _p.exists():
            raise FileNotFoundError(f'{_n} does not exist: {_p}')

    (output_dir / 'corrected').mkdir(parents=True, exist_ok=True)

    logger.info('Running XGBoost + Optuna calibration with the following configuration:\n'
                f'High-depth input file: {hi_input_path}\n'
                f'Low-depth input file: {li_input_path}\n'
                f'Output directory: {output_dir}\n'
                f'Optuna trials: {n_trials}\n'
                f'Timeout: {timeout} seconds\n'
                f'CPUs: {cpus}; GPUs: {gpus}\n')
    logger.info(f'The database of optuna HPO will be saved in {output_dir / "optuna.sqlite3"}\n'
                f'The model will be saved as "xgb_booster.json" and "xgb_booster.pkl" in {output_dir}\n'
                f'The corrected low-depth slices will be temporarily saved in {output_dir / "corrected"}\n'
                f'The final corrected low-depth file will be saved as {li_input_path.name} in {output_dir}\n')
    logger.info('==================================================')

    # Load high-depth training data
    hd_df: pd.DataFrame = read_df(f_path=hi_input_path, split=True, train_test='train')
    logger.info(f'Loaded high-depth dataset with {hd_df.shape[0]} rows.')

    # Prepare training data
    features = [i for i in all_features if i in hd_df.columns]
    X: pd.DataFrame = hd_df[features]
    y: pd.Series = hd_df['actual_beta']

    # Deal with categorical features
    for col in CAT_FEATURES:
        X[col] = X[col].astype('category')
    for col in BOOL_FEATURES:
        X[col] = X[col].astype('bool')

    timestamp: str = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger.info(f'Start training at {timestamp}')
    logger.info(f'Model will be saved in {output_dir}')

    # start training
    logger.info('Training quantile regression model for median (0.5 quantile)')
    model: xgb.Booster = train_xgb_model(X_train=X, y_train=y,
                                         n_trials=n_trials, timeout=timeout,
                                         output_dir=output_dir.as_posix(),
                                         cpus=cpus, gpus=gpus)
    # save the model
    model.save_model((output_dir / 'xgb_booster.json').as_posix())
    with open((output_dir / 'xgb_booster.pkl'), 'wb+') as f:
        pickle_dump(model, f)

    logger.info('Training completed')

    # calibrate low-depth data
    md_df = read_df(f_path=li_input_path, split=True, train_test='test')
    # Deal with categorical features
    for col in CAT_FEATURES:
        md_df[col] = md_df[col].astype('category')
    for col in BOOL_FEATURES:
        md_df[col] = md_df[col].astype('bool')

    # predict in slices
    for slice_index, slice_df in enumerate(tqdm(slice_dataframe(df=md_df))):
        with logging_redirect_tqdm():
            logger.info(f'Processing slice No.{slice_index}')

        dtest = xgb.QuantileDMatrix(slice_df[features],
                                    ref=xgb.QuantileDMatrix(X, y, nthread=cpus,
                                                            enable_categorical=True),
                                    enable_categorical=True)
        predictions = model.predict(dtest)
        slice_df['actual_beta'] = predictions

        (str_cat(df=slice_df,
                 columns=CAT_FEATURES,
                 choice='cat2str')
         .to_parquet(output_dir / 'corrected' / f'{slice_index}.parquet.lz4',
                     engine='pyarrow', compression='lz4'))

    logger.info(f'Corrected slices saved to: {output_dir / "corrected"}')

    # merge all slices
    slice_paths: list[Path] = sorted(list((output_dir / 'corrected').glob('*.parquet.lz4')))
    if not slice_paths:
        raise FileNotFoundError(f'No corrected slices found in: {output_dir / "corrected"}')

    (pl.concat([pl.scan_parquet(slice_path)
                for slice_path in slice_paths])
       .sink_parquet(output_dir / li_input_path.name,
                     compression='lz4'))
    logger.info(f'Calibration results saved to: {output_dir / li_input_path.name}')

    # clean temp files
    for slice_path in slice_paths:
        slice_path.unlink()

    logger.info('XGBoost + Optuna calibration completed successfully.')


if __name__ == '__main__':
    main()
