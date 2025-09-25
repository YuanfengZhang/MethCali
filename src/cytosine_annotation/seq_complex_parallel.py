from argparse import ArgumentParser
from concurrent.futures import as_completed, ProcessPoolExecutor
from functools import cache
from sys import path
path.extend(['./src'])

import pandas as pd  # noqa: E402
from seq_complexity import bwt_compress  # noqa: E402
from shannon import shannon_entropy  # noqa: E402


@cache
def bwt_cached(seq: str) -> float:
    return bwt_compress(seq)


@cache
def shannon_cached(seq: str) -> float:
    return shannon_entropy(seq)


def process_single_df(input: str, flank: int, complex_ls: list[int]):
    df: pd.DataFrame
    df = pd.read_table(input, header=None,
                       names=['chrom', 'start', 'end', 'seq'])
    for complex in complex_ls:
        if flank != complex:
            df[f'comp_seq_{complex}'] = df['seq'].str.slice(start=flank - complex,
                                                            stop=flank + complex + 2)
            df[f'BWT_ratio_{complex}'] = df[f'comp_seq_{complex}'].apply(bwt_cached)
            df[f'ShannonEntropy_{complex}'] = df[f'comp_seq_{complex}'].apply(shannon_cached)
            df = df.drop(f'comp_seq_{complex}', axis='columns')
        else:
            df[f'BWT_ratio_{complex}'] = df['seq'].apply(bwt_cached)
            df[f'ShannonEntropy_{complex}'] = df['seq'].apply(shannon_cached)

    df.to_csv(input.replace('.seq', '.complex'), sep='\t', index=False, header=False)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-i', '--input_list', type=str, required=True,
                            help='the list of file paths of tmp .seq files')
    arg_parser.add_argument('-k', '--flank', type=int, required=True,
                            help='the length of flank')
    arg_parser.add_argument('-c', '--complex_ls', type=str, required=True,
                            help='the list of length param of complex seq')
    arg_parser.add_argument('-p', '--parallel', type=int, required=True,
                            help='the threads to use for parallel processing')

    args = arg_parser.parse_args()
    tmp_files = args.input_list.split(',')
    flank = args.flank
    complex_ls = [int(_c) for _c in args.complex_ls.split(',')]
    parallel = args.parallel

    with ProcessPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(process_single_df, input_path, flank, complex_ls): input_path
            for input_path in tmp_files}
        for future in as_completed(futures):
            input_path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f'⚠️ Error during processing {input_path}: {e}')


if __name__ == '__main__':
    main()
