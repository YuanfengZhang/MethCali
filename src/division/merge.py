from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
import polars as pl


"""
python src/division/merge.py \
  -oi \
  -ci \
  -li \
  -hi \
  -o
"""

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

RESULT_COLUMNS: list[str] = ['chrom', 'start', 'end', 'depth', 'predicted_beta', 'actual_beta']
FULL_COLUMNS: list[str] = ['chrom', 'start', 'end', 'u', 'm', 'depth', 'predicted_beta', 'actual_beta']


def main():
    arg_parser: ArgumentParser = ArgumentParser(description='Merge all parquets after calibration.')
    arg_parser.add_argument('-oi', '--origin-input', dest='oi', type=Path, required=True,
                            help='original file used for division')
    arg_parser.add_argument('-ci', '--control-input', dest='ci', type=Path, required=True,
                            help='control file')
    arg_parser.add_argument('-li', '--low-depth-input', dest='li', type=Path, required=True,
                            help='Calibrated low depth file')
    arg_parser.add_argument('-hi', '--high-depth-input', dest='hi', type=Path, required=True,
                            help='Calibrated high depth file')
    arg_parser.add_argument('-o', '--output', dest='output', type=Path, required=True,
                            help='Output merged file')
    args: Namespace = arg_parser.parse_args()
    origin_input: Path = args.oi
    control_input: Path = args.ci
    low_input: Path = args.li
    high_input: Path = args.hi
    output: Path = args.output

    for _p, _n in {origin_input: 'original file',
                   control_input: 'control file',
                   low_input: 'low file',
                   high_input: 'high file'}.items():
        if not _p.exists():
            raise FileNotFoundError(f'{_n} does not exist: {_p}')

    ci: pl.LazyFrame
    ci = (pl.scan_parquet(control_input)
            .rename({'beta': 'predicted_beta'})
            .with_columns(pl.col('predicted_beta').alias('actual_beta'))
            .select(RESULT_COLUMNS))

    li: pl.LazyFrame
    hi: pl.LazyFrame

    (li,
     hi) = ([pl.scan_parquet(p)
               .select(RESULT_COLUMNS)
               .with_columns(pl.col('actual_beta').cast(pl.Float64))
             for p in (low_input, high_input)])

    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f'{output.name} merging')

    df: pl.DataFrame

    df = (pl.concat([ci, li, hi])
            .sort(by=['chrom', 'start'])
            .join(other=pl.scan_parquet(origin_input)
                          .select(['chrom', 'start', 'end', 'm', 'u']),
                  on=['chrom', 'start', 'end'], how='left')
            .with_columns(pl.when(pl.col('actual_beta') > 100)
                            .then(pl.lit(100.0))
                            .when(pl.col('actual_beta') < 0)
                            .then(pl.lit(0.0))
                            .otherwise(pl.col('actual_beta'))
                            .alias('actual_beta'))  # prevent illegal beta values
            .select(FULL_COLUMNS)
            .collect())

    df.write_parquet(output)

    logger.info(f'{output.name} merged successfully')


if __name__ == '__main__':
    main()
