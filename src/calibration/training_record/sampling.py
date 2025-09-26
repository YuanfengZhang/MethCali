from argparse import ArgumentParser, Namespace
from functools import reduce
import gc
import logging
from pathlib import Path
import polars as pl


"""
python src/autogluon/sampling.py plan \
  -r quartet_reference/single_c/ensembl/final \
  -i best_pipeline \
  -p /cold_data/zhangyuanfeng/methylation/modeling/plan \
  -d /cold_data/zhangyuanfeng/methylation/modeling/dataset

python src/autogluon/sampling.py run \
  -i best_pipeline \
  -p /cold_data/zhangyuanfeng/methylation/modeling/plan \
  -d /cold_data/zhangyuanfeng/methylation/modeling/dataset \
  -t /cold_data/zhangyuanfeng/methylation/modeling/tmp \
  -m /mnt/eqa/zhangyuanfeng/methylation/best_pipeline/distribution_stat/depth_10/features.parquet.lz4
"""
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

REF_LABELS: list[str] = ['D5', 'D6', 'F7', 'M8', 'BC', 'BL']

SLICE_PARAMS: dict[str, int] = {
    'b5': 0, 'b4': 1, 'b3': 2, 'b2': 3, 'b1': 4,
    'a1': 6, 'a2': 7, 'a3': 8, 'a4': 9, 'a5': 10}


def plan_sampling(label: str, ref_df: pl.DataFrame,
                  input_files: list[Path],
                  plan_dir: Path,
                  sample_size: int):
    lab_dfs: list[pl.LazyFrame] = []
    labs: list[str] = []
    for f in input_files:
        lab_df: pl.LazyFrame

        lab: str = f.name.split('_')[0]

        deph_lower_bound: int = 10
        depth_upper_bound: int = 250 if 'RR' in lab else 150

        lab_df = (pl.scan_parquet(f)
                    .with_columns(pl.lit(True).alias(lab))
                    .filter(pl.col('depth') <= depth_upper_bound,
                            pl.col('depth') > deph_lower_bound)
                    .select('chrom', 'start', 'end', lab))
        lab_dfs.append(lab_df)
        labs.append(lab)
    logger.info(f'{label}: Found {len(labs)} labs:\n{sorted(labs)}')

    label_df: pl.DataFrame
    label_df = (reduce(lambda x, y: x.join(y,
                                           on=['chrom', 'start', 'end'],
                                           how='left'),
                       [ref_df.lazy()] + lab_dfs)
                .with_columns(pl.col('beta_pyro')
                                .cut(breaks=list(range(5, 100, 5)),
                                     labels=[f'{i}-{i + 5}' for i in range(0, 100, 5)])
                                .alias('beta_bin'),
                              pl.sum_horizontal([pl.col(lab).cast(pl.Int64) for lab in labs]).alias('occurrence'))
                .filter(pl.col('occurrence') > 0)
                .with_columns([pl.col(lab).fill_null(False) for lab in labs])
                .collect())
    logger.info(f'{label}: total label dataframe shape: {label_df.shape}\n'
                f'with total {label_df["occurrence"].sum()} occurrences')

    # save the gold_train and gold_test samping list first
    gold_ref: pl.DataFrame
    gold_ref = (label_df.filter(pl.col('stratum') == 'gold')
                        .drop('stratum')
                        .sample(fraction=1, shuffle=True))
    gold_train = gold_ref.head(n=round(gold_ref.shape[0] * 0.8))
    gold_test = gold_ref.tail(n=round(gold_ref.shape[0] * 0.2))

    (gold_train.drop('beta_bin', 'occurrence')
               .write_parquet(plan_dir / f'{label}_gold_train.parquet.lz4',
                              compression='lz4'))
    logger.info(f'{label}: gold_train: {gold_train.shape[0]} cytosines,\n'
                f'plan saved to {plan_dir.as_posix()}/{label}_gold_train.parquet.lz4')
    (gold_test.drop('beta_bin', 'occurrence')
              .write_parquet(plan_dir / f'{label}_gold_validate.parquet.lz4',
                             compression='lz4'))
    logger.info(f'{label}: gold_test: {gold_test.shape[0]} cytosines,\n'
                f'plan saved to {plan_dir.as_posix()}/{label}_gold_validate.parquet.lz4')
    del gold_ref, gold_train, gold_test
    gc.collect()

    sampled_df: pl.DataFrame
    sampled_tmp_dfs: list[pl.DataFrame] = []
    for _df in (label_df.filter(pl.col('stratum') == 'silver')
                        .drop('stratum')
                        .partition_by('beta_bin', 'occurrence', include_key=False)):
        if _df.shape[0] < sample_size:
            sampled_tmp_dfs.append(_df)
        else:
            sampled_tmp_dfs.append(_df.sample(n=sample_size))

    sampled_df: pl.DataFrame = pl.concat(sampled_tmp_dfs)

    logger.info(f'{label}: sampled {sampled_df.shape[0]} cytosines from silver stratum '
                'targeting beta_bin and occurrence columns were extracted')

    del label_df, sampled_tmp_dfs
    gc.collect()

    sampled_dfs: list[pl.DataFrame]

    per_k_size = len(sampled_df) // 5
    sampled_dfs = (sampled_df.sample(fraction=1, shuffle=True)
                             .with_row_index('idx')
                             .with_columns(pl.col('idx') // per_k_size)
                             .partition_by(by='idx', include_key=False))[: 5]
    logger.info(f'{label}: sampled 5 dataframes, each with {per_k_size} cytosines')

    for idx, df in enumerate(sampled_dfs):
        df.write_parquet(plan_dir / f'{label}_{idx}.parquet.lz4', compression='lz4')

    logger.info(f'{label}: plan files saved to {plan_dir.as_posix()}')


def prepare_plan(args: Namespace):
    ref_dir: Path = Path(args.ref_dir)
    input_dir: Path = Path(args.input_dir)
    blacklist_patterns: list[str] | None = None if args.blacklist == '' else args.blacklist.split(',')
    plan_dir: Path = Path(args.plan_dir)
    dataset_dir: Path = Path(args.dataset_dir)
    sample_size: int = args.sample_size

    if not ref_dir.exists() or not ref_dir.is_dir():
        raise FileNotFoundError(f'ref_dir: {ref_dir.as_posix()} is not a valid directory')
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f'input_dir: {input_dir.as_posix()} is not a valid directory')

    plan_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info('All directories are valid\n'
                f'ref_dir: {ref_dir.as_posix()}\n'
                f'input_dir: {input_dir.as_posix()}\n'
                f'plan_dir: {plan_dir.as_posix()}\n'
                f'dataset_dir: {dataset_dir.as_posix()}')

    cytosine_features: list[str]
    if args.split_seq:
        cytosine_features = ['b5', 'b4', 'b3', 'b2', 'b1', 'a1', 'a2', 'a3', 'a4', 'a5',
                             'GC%_70', 'CpG_GC_ratio_70', 'GC_skew_70', 'ShannonEntropy_70', 'BWT_ratio_70',
                             'cpg', 'enhancer', 'promoter', 'location']
    else:
        cytosine_features = ['seq_5', 'GC%_70', 'CpG_GC_ratio_70', 'GC_skew_70', 'ShannonEntropy_70',
                             'BWT_ratio_70', 'cpg', 'enhancer', 'promoter', 'location']

    logger.info(f'Using cytosine features: {cytosine_features}')

    for ref_label in REF_LABELS:
        logger.info(f'Generating sampling plan for ref label: {ref_label}')
        if not (ref_dir / f'{ref_label}.parquet.lz4').exists():
            raise FileNotFoundError(f'Reference file for label {ref_label} does not exist')

        if blacklist_patterns:
            label_files: list[Path] = [
                i for i in input_dir.glob('*.parquet.lz4')
                if i.name.split('_')[1] == ref_label and all(pattern not in i.name
                                                             for pattern in blacklist_patterns)]
        else:
            label_files: list[Path] = [
                i for i in input_dir.glob('*.parquet.lz4')
                if i.name.split('_')[1] == ref_label]

        if not label_files:
            raise FileNotFoundError(f'No input files found for label {ref_label} in {input_dir.as_posix()}')
        logger.info(f'{ref_label}: Found {len(label_files)} input files')

        lazy_read: pl.LazyFrame
        if 'a1' in cytosine_features:
            lazy_read = (pl.scan_parquet(ref_dir / f'{ref_label}.parquet.lz4')
                           .filter(pl.col('in_hcr'))
                           .with_columns(pl.when(pl.col('seq_5').str.slice(5, 1) == 'C')
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
                           .select('chrom', 'start', 'end', 'beta_pyro', 'stratum', *cytosine_features))
        else:
            lazy_read = (pl.scan_parquet(ref_dir / f'{ref_label}.parquet.lz4')
                           .filter(pl.col('in_hcr'))
                           .select('chrom', 'start', 'end', 'beta_pyro', 'stratum', *cytosine_features))

        ref_df: pl.DataFrame = lazy_read.collect()
        logger.info(f'{ref_label}: ref dataframe loaded with shape: {ref_df.shape}')

        plan_sampling(label=ref_label, ref_df=ref_df, input_files=label_files,
                      plan_dir=plan_dir, sample_size=sample_size)


def run_sampling(args: Namespace):
    input_dir: Path = Path(args.input_dir)
    plan_dir: Path = Path(args.plan_dir)
    dataset_dir: Path = Path(args.dataset_dir)
    tmp_dir: Path = Path(args.tmp_dir)
    metadata_file: Path = Path(args.metadata_file)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f'input_dir: {input_dir.as_posix()} is not a valid directory')
    if not plan_dir.exists() or not plan_dir.is_dir():
        raise FileNotFoundError(f'plan_dir: {plan_dir.as_posix()} is not a valid directory')
    if not metadata_file.exists():
        raise FileNotFoundError(f'metadata_file: {metadata_file.as_posix()} does not exist')
    dataset_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    metadata: pl.DataFrame = pl.read_parquet(metadata_file)
    metadata_columns: list[str] = metadata.columns[1:]
    logger.info(f'Metadata loaded with {len(metadata_columns)} sample-level features:\n{metadata_columns}')

    for plan_file in sorted(plan_dir.glob('*.parquet.lz4')):
        fname: str = plan_file.name.split('.')[0]
        label: str = fname[: 2]

        logger.info(f'{label}: generating dataset from {fname}')

        (tmp_dir / fname).mkdir(parents=True, exist_ok=True)

        plan_df: pl.DataFrame = pl.read_parquet(plan_file)
        logger.info(f'{label}: plan dataframe loaded containing {plan_df.shape[0]} cytosines')

        labs: list[str] = [i for i in plan_df.columns if any(i.startswith(p) for p in ('BS', 'EM', 'PS', 'RR'))]
        for lab in labs:
            lab_sites: pl.DataFrame = plan_df.filter(pl.col(lab)).drop(labs)
            if lab_sites.shape[0] == 0:
                logger.warning(f'No sites found for label {label} in {lab}, skipping...')
                continue
            lab_metadata = metadata.filter(pl.col('fname') == f'{lab}_{label}_1')
            lab_sites = (lab_sites.lazy()
                                  .join(other=pl.scan_parquet(input_dir / f'{lab}_{label}_1.parquet.lz4')
                                                .select('chrom', 'start', 'end', 'beta', 'depth'),
                                        on=['chrom', 'start', 'end'], how='left')
                                  .with_columns(pl.lit(lab[:2]).alias('method'))
                                  .with_columns([pl.lit(lab_metadata[c].first()).alias(c)
                                                 for c in metadata_columns])
                                  .collect())
            lab_sites.write_parquet(tmp_dir / fname / f'{lab}.parquet.lz4', compression='lz4')

        # merge all the lab sites into one parquet file
        lab_parquets: list[Path] = list((tmp_dir / fname).glob('*.parquet.lz4'))
        if not lab_parquets:
            logger.warning(f'No lab parquets found for {fname}, skipping...')
            continue
        (pl.concat([pl.scan_parquet(p) for p in lab_parquets])
           .rename({'beta': 'predicted_beta', 'beta_pyro': 'actual_beta'})
           .sink_parquet(dataset_dir / f'{fname}.parquet.lz4'))


def main():
    arg_parser: ArgumentParser = ArgumentParser(description='Generating train and test sampling plan and datasets')
    subparsers = arg_parser.add_subparsers(dest='subcommand', required=True)
    plan_parser = subparsers.add_parser('plan', help='Generating train and test sampling plan')
    plan_parser.add_argument('-r', '--ref-dir', dest='ref_dir', type=str,
                             help='The directory containing the reference parquet files')
    plan_parser.add_argument('-i', '--input-dir', dest='input_dir', type=str,
                             help='The directory containing the formatted parquet files from best pipeline')
    plan_parser.add_argument('-b', '--blacklist', dest='blacklist', default='', type=str,
                             help='blacklist of patterns to exclude for training and testing, '
                                  'e.g., "BS1,BS2,EM1,EM3"')
    plan_parser.add_argument('-p', '--plan-dir', dest='plan_dir', type=str,
                             help='The path to the directory to store parquet files containing the sampling plan')
    plan_parser.add_argument('-d', '--dataset-dir', dest='dataset_dir', type=str,
                             help='The path to the directory to store the train and test dataset as parquet files')
    plan_parser.add_argument('--split-seq', dest='split_seq', action='store_true',
                             help='Whether to split the seq_5 column into individual bases'),
    plan_parser.add_argument('-n', '--number', dest='sample_size', type=int, default=500,
                             help='The number of samples to be sampled from every beta_bin and '
                                  'occurrence combination, default is 500')
    plan_parser.set_defaults(func=prepare_plan)

    run_parser = subparsers.add_parser('run', help='Run the sampling plan to generate datasets')
    run_parser.add_argument('-i', '--input-dir', dest='input_dir', type=str,
                            help='The directory containing the formatted parquet files from best pipeline')
    run_parser.add_argument('-p', '--plan-dir', dest='plan_dir', type=str,
                            help='The path to the directory to store parquet files containing the sampling plan')
    run_parser.add_argument('-d', '--dataset-dir', dest='dataset_dir', type=str,
                            help='The path to the directory to store the train and test dataset as parquet files')
    run_parser.add_argument('-t', '--tmp-dir', dest='tmp_dir', type=str,
                            help='The path to the directory to store the temporary dataset as parquet files')
    run_parser.add_argument('-m', '--metadata', dest='metadata_file', type=str,
                            help='The path to the parquet to store the metadata of every formatted bedgraph parquet')
    run_parser.set_defaults(func=run_sampling)

    args: Namespace = arg_parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
