from argparse import ArgumentParser, Namespace
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import logging
from pathlib import Path
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from typing import Literal
import numpy as np
import pandas as pd
from scipy.stats import entropy, gaussian_kde, skew
from scipy.signal import find_peaks


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


INFO_COLS: list[str] = ['beta', 'depth']


DEPTH_METADATA_COLS: list[str] = ['depth5_count', 'depth_std', 'depth_max',
                                  'depth_cv', 'depth_skew', 'depth_kurtosis',
                                  'depth_peak_num', 'depth_peak1',
                                  'depth_peak2', 'depth_peak3',
                                  'depth_peak4', 'depth_peak5']

BETA_METADATA_COLS: list[str] = ['depth10_count', 'beta_q5', 'beta_q6',
                                 'beta_q7', 'beta_q8', 'beta_q9',
                                 'beta_q10', 'beta_q11', 'beta_q12',
                                 'beta_q13', 'beta_q82', 'beta_q85',
                                 'beta_q86', 'beta_q87', 'beta_q88',
                                 'beta_q89', 'beta_q90', 'beta_min',
                                 'beta_max', 'beta_entropy']


def to_float(a_list) -> list[float]:
    return [float(x) if x is not None else None for x in a_list]


def get_keys(info_col: str, num_peaks: int) -> list[str]:
    count_col: str = 'depth10_count' if 'beta' in info_col else 'depth5_count'
    return ([
            f'{info_col}_q{i}' for i in range(1, 100)
        ] + [
            count_col, f'{info_col}_mean', f'{info_col}_std',
            f'{info_col}_min', f'{info_col}_max', f'{info_col}_mad',
            f'{info_col}_cv', f'{info_col}_skew', f'{info_col}_kurtosis',
            f'{info_col}_iqr', f'{info_col}_entropy', f'{info_col}_peak_num'
        ] + [
            f'{info_col}_peak{i}' for i in range(1, num_peaks + 1)
        ])


def get_quantiles(series: pd.Series) -> list[float]:
    # get 1 to 99 percentiles
    return np.percentile(series, q=np.arange(1, 100, 1)).tolist()


def get_mad(series: pd.Series,
            median: float) -> float:
    mad_value = np.median(np.abs(series - median))
    return float(mad_value)


def get_description(series: pd.Series) -> pd.Series:
    return series.describe()


def get_peaks(series: pd.Series,
              origin_col: str,
              n: int) -> tuple[int, list[float | None]]:
    # init KDE
    if 'beta' in origin_col:
        kde = gaussian_kde(series.values, bw_method=1)
    else:
        kde = gaussian_kde(series.values)

    # create linspace and calculate density
    x_vals = np.linspace(series.min(), series.max(), 1000)
    spaced_density = kde(x_vals)

    # find peaks
    if 'beta' in origin_col:
        peaks_idxs, _ = find_peaks(spaced_density, distance=10)
    elif 'depth' in origin_col:
        peaks_idxs, _ = find_peaks(spaced_density, prominence=.01)
    else:
        peaks_idxs, _ = find_peaks(spaced_density, prominence=0)

    peaks = x_vals[peaks_idxs]
    densities = spaced_density[peaks_idxs]

    sorted_peaks = peaks[np.argsort(-densities)]

    if len(sorted_peaks) >= n:
        selected_peaks = (sorted_peaks[: n]).tolist()
    else:
        selected_peaks = sorted_peaks.tolist() + [None] * (n - len(sorted_peaks))

    return len(peaks), selected_peaks


def process_series(series: pd.Series, pkl_dir: Path,
                   fname: str, method: Literal['BS', 'EM', 'PS', 'RR'],
                   info_col: str, num_peaks: int):
    output_stem: Path = pkl_dir / f'{fname}-{info_col.strip()}'

    description: pd.Series = get_description(series)
    logger.info(f'[{fname}]\t{info_col}:\ndescription: {description}')

    # get mad
    mad = get_mad(series, description['50%'])

    # get cv
    cv = description['std'] / description['mean'] if description['mean'] != 0 else np.nan

    # get skew and kurtosis
    skew_value: float = float(skew(series.to_numpy(), nan_policy='omit'))
    kurtosis_value: float = series.kurt()

    # get quantiles
    quantiles: list[float] = get_quantiles(series)
    logger.info(f'[{fname}]\t{info_col}: quantiles calculated: {quantiles}')

    # get IQR
    iqr = quantiles[74] - quantiles[24]

    # get entropy
    hist, _ = np.histogram(series, bins=20, density=True)
    entropy_value: float = entropy(hist, base=2) if np.any(hist) else 0.0

    # get peaks
    peak_num: int
    peaks: list[float]
    peak_num, peaks = get_peaks(series, info_col, num_peaks)
    logger.info(f'[{fname}]\t{info_col}: {peak_num} peaks found, best {num_peaks} at {peaks}')

    # save metadata
    with open(output_stem.with_suffix('.pkl'), 'wb') as f:
        pickle_dump({
            'method': method,
            'quantiles': quantiles,
            'description': [
                description.tolist()[i] for i in (0, 1, 2, 3, 7)
            ] + [
                mad, cv, skew_value, kurtosis_value, iqr, entropy_value, peak_num
            ],
            'peaks': peaks}, f)
    logger.info(f'[{fname}]\t{info_col}: quantiles and peaks saved to '
                f'{output_stem.with_suffix(".pkl").as_posix()}')


def run_stat(args: Namespace):
    input_path: Path = Path(args.input)
    logger.info(f'Parsed arguments: {args}')

    if input_path.exists() and input_path.is_file():
        pass
    else:
        raise FileNotFoundError(f'Input file {input_path} does not exist.')

    fname: str = input_path.name.split('.')[0]

    pkl_dir: Path = Path(args.pkl_dir)
    pkl_dir.mkdir(parents=True, exist_ok=True)

    method: Literal['BS', 'EM', 'PS', 'RR'] = args.method

    logger.info(f'[{fname}] Run Config:\n'
                f'--input path:\t{input_path.as_posix()}\n'
                f'--output directory:\t{pkl_dir.as_posix()}\n'
                f'--sequencing method:\t{method}\n'
                '=====================================================================')

    df: pd.DataFrame = pd.read_parquet(input_path)
    logger.info(f'[{fname}] Loaded data from {input_path} with shape {df.shape}')

    for info_col in INFO_COLS:
        if info_col not in df.columns:
            raise ValueError(f'Column {info_col} not found in input file {input_path}.')

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures: list[Future[None]] = [
            executor.submit(process_series,
                            df[info_col].copy(deep=True),
                            pkl_dir, fname, method,
                            info_col, args.num_peaks)
            for info_col in INFO_COLS
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f'[{fname}] Error occurred while processing: {e}')

    beta_pkl_f: Path = pkl_dir / f'{fname}-beta.pkl'
    depth_pkl_f: Path = pkl_dir / f'{fname}-depth.pkl'

    with open(beta_pkl_f, 'rb') as f:
        beta_data = pickle_load(f)
    with open(depth_pkl_f, 'rb') as f:
        depth_data = pickle_load(f)

    if len(beta_data['peaks']) < args.num_peaks:
        beta_peaks: list[float] = beta_data['peaks'] + [-1.0] * (args.num_peaks - len(beta_data['peaks']))
    else:
        beta_peaks = beta_data['peaks'][:args.num_peaks]

    beta_peaks = [i if i else -1.0 for i in beta_peaks]

    if len(depth_data['peaks']) < args.num_peaks:
        depth_peaks: list[float] = depth_data['peaks'] + [-1.0] * (args.num_peaks - len(depth_data['peaks']))
    else:
        depth_peaks = depth_data['peaks'][:args.num_peaks]

    depth_peaks = [i if i else -1.0 for i in depth_peaks]

    beta_dict: dict[str, float] = dict(
        zip(
            get_keys(info_col='beta', num_peaks=args.num_peaks),
            to_float(beta_data['quantiles']) + to_float(beta_data['description']) + to_float(beta_peaks)
        )
    )

    depth_dict: dict[str, float] = dict(
        zip(
            get_keys(info_col='depth', num_peaks=args.num_peaks),
            to_float(depth_data['quantiles']) + to_float(depth_data['description']) + to_float(depth_peaks)
        )
    )

    # extract only the columns used in the model
    metadata_dict = {'method': method}

    for metadata_col in BETA_METADATA_COLS:
        metadata_dict[metadata_col] = beta_dict[metadata_col]

    for metadata_col in DEPTH_METADATA_COLS:
        metadata_dict[metadata_col] = depth_dict[metadata_col]

    with open(pkl_dir / f'{fname}-metadata.pkl', 'wb') as f:
        pickle_dump(metadata_dict, f)

    logger.info(f'[{fname}] Metadata for calibration saved to {pkl_dir / f"{fname}-metadata.pkl"}')


def run_merge(args: Namespace):
    pkl_dir: Path = Path(args.pkl_dir)
    if not pkl_dir.exists() or not pkl_dir.is_dir():
        raise FileNotFoundError(f'Output directory {pkl_dir} does not exist.')

    pkl_files: list[Path] = list(pkl_dir.glob('*.pkl'))
    if not pkl_files:
        raise FileNotFoundError(f'No .pkl files found in {pkl_dir.as_posix()}')

    merged_data: list[list[str | float | int]] = []
    for pkl_file in pkl_files:
        fname: str = pkl_file.parent.name
        info_col: str = pkl_file.stem
        with open(pkl_file, 'rb') as f:
            data = pickle_load(f)
            peaks: list[float | None] = data['peaks']

            if len(peaks) < args.num_peaks:
                padded_peaks: list[float | None] = peaks + [None] * (args.num_peaks - len(peaks))
            else:
                padded_peaks = peaks[:args.num_peaks]

            merged_data.append([
                fname, data['method'], info_col
            ] + to_float(data['quantiles'])
              + to_float(data['description'])
              + to_float(padded_peaks))

    df: pd.DataFrame = pd.DataFrame(
        merged_data,
        columns=[
            'fname', 'method', 'info_col'
        ] + [
            f'q{i}' for i in range(1, 100)
        ] + [
            'count', 'mean', 'std', 'min', 'max', 'mad', 'cv',
            'skew', 'kurtosis', 'iqr', 'entropy', 'peak_num',
        ] + [
            f'peak{i}' for i in range(1, args.num_peaks + 1)
        ])
    output_path: Path = Path(args.output_stem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(f'{output_path}.csv'), index=False)
    df.to_parquet(Path(f'{output_path}.parquet'))
    logger.info(f'Merged data saved to {output_path.as_posix()}.csv and .parquet')


def main():
    arg_parser: ArgumentParser = ArgumentParser(description='Plot and save the distribution '
                                                            'of info cols in formatted bedgraph')
    subparsers = arg_parser.add_subparsers(dest='subcommand', required=True)
    stat_parser: ArgumentParser = subparsers.add_parser('stat',
                                                        help='Perform statistical analysis and plotting')
    stat_parser.add_argument('-i', '--input', dest='input',
                             type=str, required=True,
                             help='Input file path (formatted bedgraph)')
    stat_parser.add_argument('-o', '--pkl-dir', dest='pkl_dir',
                             type=str, required=True,
                             help='Root Output directory for statistics')
    stat_parser.add_argument('-m', '--method', dest='method',
                             choices={'BS', 'EM', 'PS', 'RR'},
                             help='sequencing method for this sample')
    stat_parser.add_argument('-n', '--num-peaks', dest='num_peaks',
                             type=int, default=5, help='Number of peaks to find')
    stat_parser.add_argument('-p', '--parallel', dest='parallel',
                             type=int, default=2, help='Number of parallel processes to use')

    stat_parser.set_defaults(func=run_stat)

    merge_parser: ArgumentParser = subparsers.add_parser('merge',
                                                         help=('Merge all .pkl files into a single CSV. '
                                                               'It is only for statistics.'))
    merge_parser.add_argument('-p', '--pkl-dir', dest='pkl_dir', type=str, required=True,
                              help='Directory to search for .pkl files')
    merge_parser.add_argument('-o', '--output-stem', dest='output_stem', type=str, required=True,
                              help='Stem for output files, including the directory. E.g., /a/b/c/metadata')
    merge_parser.add_argument('-n', '--num-peaks', dest='num_peaks',
                              type=int, default=5, help='Number of peaks to find')
    merge_parser.set_defaults(func=run_merge)

    args = arg_parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
