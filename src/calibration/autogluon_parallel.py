from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def slice_dataframe(df: pd.DataFrame, slice_size: int = 100_000) -> list[pd.DataFrame]:
    num_chunks: int = int(np.ceil(len(df) / slice_size))
    return [df.iloc[i * slice_size: (i + 1) * slice_size] for i in range(num_chunks)]


def main():
    arg_parser: ArgumentParser = ArgumentParser(description='Run AutoGluon calibration on a dataset.')
    arg_parser.add_argument('-i', '--input-dir', type=Path, required=True, help='Input dataset directory.')
    arg_parser.add_argument('-o', '--output-dir', type=Path, required=True, help='Output parquet directory.')
    arg_parser.add_argument('-w', '--whitelist', type=str, default='',
                            help='Whitelist of input files to process, comma-separated.')
    arg_parser.add_argument('-md', '--model-dir', type=Path, required=True,
                            help='Directory containing the trained AutoGluon model.')
    arg_parser.add_argument('-mn', '--model-name', type=str, default='LightGBMXT_BAG_L2',
                            help=('Name of the AutoGluon model to use for calibration. '
                                  'Default is "LightGBMXT_BAG_L2". Other available models '
                                  'can be found in the leaderboard.csv in the model directory.'))

    args: Namespace = arg_parser.parse_args()
    output_dir: Path = args.output_dir
    model_dir: Path = args.model_dir
    model_name: str = args.model_name
    whitelist: list[str] = args.whitelist.split(',') if args.whitelist else []

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    parquet_io_paths: dict[Path, Path] = {
        p: output_dir / p.name for p in args.input_dir.glob('*.parquet.lz4')
        if p.is_file() and not (output_dir / p.name).exists()
    }

    if whitelist:
        parquet_io_paths = {p: output_dir / p.name for p in parquet_io_paths.keys()
                            if any(w in p.name for w in whitelist)}

    if not (model_dir / 'predictor.pkl').exists():
        raise FileNotFoundError(f'AutoGluon TabularPredictor model does not exist: {model_dir / "predictor.pkl"}')

    logger.info('Running AutoGluon calibration with the following configuration:\n'
                '==================================================\n'
                f'Number of input files to handle: {len(parquet_io_paths)}\n'
                f'Output parquet directory: {output_dir}\n'
                f'Model directory: {model_dir}\n'
                '==================================================\n')

    predictor: TabularPredictor = TabularPredictor.load(path=model_dir.as_posix(),
                                                        verbosity=3,
                                                        require_py_version_match=False)
    predictor.compile(models=[model_name], compiler_configs={'GBM': {'compiler': 'native'}})

    for input_path, output_path in tqdm(parquet_io_paths.items()):
        with logging_redirect_tqdm():
            logger.info(f'Processing input file: {input_path}')
            input_df: pd.DataFrame = pd.read_parquet(input_path)  # type: ignore

            logger.info(f'Loaded input dataset with {input_df.shape[0]} rows.')

            input_df['actual_beta'] = predictor.predict(data=input_df,
                                                        model=model_name,
                                                        as_pandas=True)
            input_df.to_parquet(output_path, engine='pyarrow', compression='lz4')

        #     slices: list[pd.DataFrame] = slice_dataframe(input_df)

        #     for idx, slice in enumerate(slices):
        #         slice['actual_beta'] = predictor.predict(data=slice,
        #                                                  model=model_name,
        #                                                  as_pandas=True)
        #         slice.to_parquet(f'{output_path}.{idx}',
        #                          engine='pyarrow', compression='lz4')

        # (pl.concat([pl.scan_parquet(f'{output_path}.{idx}')
        #             for idx in range(len(slices))])
        #    .sink_parquet(output_path, compression='lz4'))
        # logger.info(f'Calibration results saved to: {output_path}')

        # for p in [Path(f'{output_path}.{idx}') for idx in range(len(slices))]:
        #     p.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
    logger.info('AutoGluon calibration completed successfully.')
