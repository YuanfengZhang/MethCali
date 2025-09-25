from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
from autogluon.tabular import TabularPredictor
import pandas as pd


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    arg_parser: ArgumentParser = ArgumentParser(description='Run AutoGluon calibration on a dataset.')
    arg_parser.add_argument('-i', '--input', type=Path, required=True, help='Input dataset file.')
    arg_parser.add_argument('-o', '--output', type=Path, required=True, help='Output parquet file.')
    arg_parser.add_argument('-md', '--model-dir', type=Path, required=True,
                            help='Directory containing the trained AutoGluon model.')
    arg_parser.add_argument('-mn', '--model-name', type=str, default='LightGBMXT_BAG_L1',
                            help=('Name of the AutoGluon model to use for calibration. '
                                  'Default is "LightGBMXT_BAG_L1". Other available models '
                                  'can be found in the leaderboard.csv in the model directory.'))

    args: Namespace = arg_parser.parse_args()
    input_path: Path = args.input
    output_parquet: Path = args.output
    model_dir: Path = args.model_dir
    model_name: str = args.model_name

    for _p, _n in {input_path: 'Input dataset file',
                   model_dir / 'predictor.pkl': 'AutoGluon TabularPredictor model'}.items():
        if not _p.exists():
            raise FileNotFoundError(f'{_n} does not exist: {_p}')

    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    logger.info('Running AutoGluon calibration with the following configuration:\n'
                '==================================================\n'
                f'Input file: {input_path}\n'
                f'Output parquet file: {output_parquet}\n'
                f'Model directory: {model_dir}\n'
                '==================================================\n')

    input_df: pd.DataFrame = pd.read_parquet(input_path)
    logger.info(f'Loaded input dataset with {input_df.shape[0]} rows.')

    predictor: TabularPredictor = TabularPredictor.load(path=model_dir.as_posix(),
                                                        verbosity=3,
                                                        require_py_version_match=False)

    input_df['actual_beta'] = predictor.predict(data=input_df,
                                                model=model_name,
                                                as_pandas=True)

    input_df.to_parquet(output_parquet, engine='pyarrow', compression='lz4')
    logger.info(f'Calibration results saved to: {output_parquet}')


if __name__ == '__main__':
    main()
    logger.info('AutoGluon calibration completed successfully.')
