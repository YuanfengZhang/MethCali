# -*- coding: utf-8 -*-
# @Author: Zhang Yuanfeng
# @Email: zhangyuanfeng1997@foxmail.com
# @Last modified time: 2024-07-21
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter, ArgumentTypeError
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass, field
import gc
import logging
from pathlib import Path
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from subprocess import DEVNULL, PIPE, run
from tempfile import NamedTemporaryFile
from typing import Literal
import warnings
import polars as pl
from zstandard import ZstdCompressor


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', message='Polars found a filename')


DESCRIPTION_MESSAGE = (
    'To use this file, run cmdln like this:\n'
    "time python src/annotate.py \\\n    "
    "-i formatted/lab1_sample1_rep1.parquet.lz4 \\\n    "
    "-o annotated_cytosines \\\n    "
    "-t tmp \\\n    "
    "-r ref/GRCh38.fa \\\n    "
    "-e genomic_tools \\\n    "
    "-m conda_run \\\n    "
    "--annotatr-image annotatr:1.33.0 \\\n    "
    "-s src/annotatr_bed_parallel.R \\\n    "
    "-x src/seq_complexity.py \\\n    "
    "--cutoff 3 \\\n    "
    "-s src/annotatr_bed_parallel.R\n\n"
    "the conda env to run this python script should contain polars and zstandard,\n"
    "To create:\n1. mamba create -n data python=3.12 zstandard \n"
    "2. mamba activate data\n3. pip install 'polars[all]'\n\n"
    "the conda env for -e param should contain bedtools.\n"
    "To create:\n1. mamba create -n genomic_tools -c bioconda zstandard bedtools"
)

# INDIVIDUAL REFERENCE FASTA FILE
# FASTA_DIR: str = '/hot_warm_data/ref/quartet/DNA/custom_genome'
# REF_FASTA_MAP: dict[str,
#                     str] = {'D5': f'{FASTA_DIR}/D5.fa',
#                             'D6': f'{FASTA_DIR}/D5.fa',
#                             'T1': f'{FASTA_DIR}/D5.fa',
#                             'T2': f'{FASTA_DIR}/D5.fa',
#                             'T3': f'{FASTA_DIR}/D5.fa',
#                             'T4': f'{FASTA_DIR}/D5.fa',
#                             'F7': f'{FASTA_DIR}/F7.fa',
#                             'M8': f'{FASTA_DIR}/M8.fa',
#                             'BC': f'{FASTA_DIR}/BC.fa',
#                             'BL': f'{FASTA_DIR}/BL.fa',
#                             'HF': f'{FASTA_DIR}/BL.fa'}

# CHROMOSOMES
CHROMOSOMES: list[str] = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5',
                          'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
                          'chr11', 'chr12', 'chr13', 'chr14',
                          'chr15', 'chr16', 'chr17', 'chr18',
                          'chr19', 'chr20', 'chr21', 'chr22',
                          'chrX', 'chrY']

# ASSIGN THE FLANK TO ANNOTATE DIRECTLY AND CALCULATE GC%.
ANNO_FLANK_LS: list[int] = [5]
GC_FLANK_LS: list[int] = [70]
COMPLEXITY_FLANK_LS: list[int] = [70]

to_retrieve = max(*ANNO_FLANK_LS, *GC_FLANK_LS, *COMPLEXITY_FLANK_LS)

# Python script to get sequence complexity.
SEQ_COMPLEX_SCRIPT = Path('/mnt/eqa/zhangyuanfeng/methylation/'
                          'quartet_reference/truset_1120/src/'
                          'seq_complex_parallel.py')

# R script to get annotation.
ANNOTATR_SCRIPT = Path('/mnt/eqa/zhangyuanfeng/methylation/'
                       'quartet_reference/truset_1120/src/annotatr.R')

# annotation map
ANNO_REPLACE_DICT: dict[str,
                        str] = ({
                            'hg38_cpg_islands': 'cpg_island', 'hg38_cpg_shores': 'cpg_shore',
                            'hg38_cpg_shelves': 'cpg_shelve', 'hg38_cpg_inter': 'cpg_inter',
                            'hg38_genes_promoters': 'promoter', 'hg38_enhancers_fantom': 'enhancer',
                            'hg38_lncrna_gencode': 'lcrna', 'hg38_genes_1to5kb': 'TTS_1to5kb',
                            'hg38_genes_5UTRs': '5UTR', 'hg38_genes_3UTRs': '3UTR',
                            'hg38_genes_firstexons': '1st_exon',
                            'hg38_genes_intronexonboundaries': 'intron_exon_bound',
                            'hg38_genes_exonintronboundaries': 'exon_intron_bound',
                            'hg38_genes_introns': 'intron', 'hg38_genes_cds': 'cds',
                            'hg38_genes_exons': 'exon', 'hg38_genes_intergenic': 'intergenic'
                        })


@dataclass
class AnnotationMap:
    name: str
    anno2order: dict[str, int]
    order2anno: dict[int, str] | dict[int, bool]
    default_order: int
    default_anno: str | bool
    dtype: pl._typing.PolarsDataType = field(init=False)

    def __post_init__(self):
        if isinstance(self.default_anno, str):
            self.dtype = pl.String
        else:
            self.dtype = pl.Boolean


promoter_mapping = AnnotationMap(name='promoter',
                                 anno2order={'promoter': 0},
                                 order2anno={0: True, 1: False},
                                 default_order=1,
                                 default_anno=False)
enhancer_mapping = AnnotationMap(name='enhancer',
                                 anno2order={'enhancer': 0},
                                 order2anno={0: True, 1: False},
                                 default_order=1,
                                 default_anno=False)
lncrna_mapping = AnnotationMap(name='lncrna',
                               anno2order={'lncrna': 0},
                               order2anno={0: True, 1: False},
                               default_order=1,
                               default_anno=False)
cpg_mapping = AnnotationMap(name='cpg',
                            anno2order={'cpg_island': 0, 'cpg_shore': 1,
                                        'cpg_shelve': 2, 'cpg_inter': 3},
                            order2anno={0: 'cpg_island', 1: 'cpg_shore',
                                        2: 'cpg_shelve', 3: 'cpg_inter'},
                            default_order=3,
                            default_anno='cpg_inter')
location_mapping = AnnotationMap(name='location',
                                 anno2order={'TTS_1to5kb': 0, '5UTR': 1,
                                             '3UTR': 2, '1st_exon': 3,
                                             'intron': 4, 'intron_exon_bound': 5,
                                             'exon_intron_bound': 6,
                                             'cds': 7, 'exon': 8, 'intergenic': 9},
                                 order2anno={0: 'TTS_1to5kb', 1: '5UTR', 2: '3UTR',
                                             3: '1st_exon', 4: 'intron',
                                             5: 'intron_exon_bound', 6: 'exon_intron_bound',
                                             7: 'cds', 8: 'exon', 9: 'intergenic'},
                                 default_order=9,
                                 default_anno='intergenic')


def validate_docker_image(docker_image: str) -> str:
    result = run(f'docker inspect --type=image {docker_image}',
                 shell=True, stdout=PIPE, stderr=DEVNULL, text=True)
    stdout = result.stdout.strip()

    image_exists = False

    if stdout and stdout != '[]' and 'Error' not in stdout:
        image_exists = True

    if not image_exists:
        raise ArgumentTypeError(f'Docker image {docker_image} not found')

    return docker_image


def validate_conda_env(conda_env: str) -> str:
    env_exists = False
    if Path(conda_env).is_dir():
        env_exists = True
    else:
        cmdln: str
        if '/' in conda_env:
            cmdln = f'conda run -p {conda_env} bedtools --version'
        else:
            cmdln = f'conda run -n {conda_env} bedtools --version'
        result = run(cmdln,
                     shell=True, stdout=PIPE, stderr=DEVNULL, text=True)
        if result.returncode == 0 and 'bedtools v2' in result.stdout.strip():
            env_exists = True

    if not env_exists:
        raise ArgumentTypeError(f'Conda environment {conda_env} not found')

    return conda_env


def generate_bed(df: pl.DataFrame,
                 tmp_dir: Path,
                 retrieve_flank: int = 75,
                 row_limit: int = 1000000,
                 flank: bool = True) -> list[str]:
    to_slice: pl.DataFrame

    if flank:
        to_slice = (df.with_columns([(pl.col('start') - retrieve_flank),
                                     (pl.col('end') + retrieve_flank)])
                      .select(['chrom', 'start', 'end']))
    else:
        to_slice = df.select(['chrom', 'start', 'end'])

    tmp_file_list: list[str] = []

    for sliced_df in to_slice.iter_slices(n_rows=row_limit):
        with NamedTemporaryFile(delete=False, delete_on_close=False,
                                dir=tmp_dir, suffix='.bed') as tmp_file:
            sliced_df.write_csv(tmp_file.name, include_header=False,
                                separator='\t', line_terminator='\n')
            tmp_file_list.append(tmp_file.name)
            tmp_file.close()

    del to_slice
    gc.collect()

    return tmp_file_list


def get_seq_info(fasta_file: Path, bed_file: str,
                 conda_env: str,
                 conda_method: Literal['conda_run',
                                       'source_activate'] = 'conda_run'):
    # Define the command line to be executed
    _cmdln: str
    _cmdln = (f'bedtools getfasta -fi {fasta_file} -bed {bed_file}'
              f' -bedOut >{bed_file}.seq')
    _cmdln = (f'conda run -n {conda_env} ' + _cmdln
              if conda_method == 'conda_run'
              else f'source activate {conda_env} && ' + _cmdln)

    # Execute the command line
    run(_cmdln, shell=True)
    print(f'finished: {bed_file}')


def retrieve_sequence(run_name: str, bed_file_list: list[str],
                      fasta_file: Path,
                      tmp_dir: str, conda_env: str,
                      complexity_script: Path,
                      parallel: int = 8,
                      conda_method: Literal['conda_run',
                                            'source_activate'] = 'conda_run'):
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        tasks = (
            [executor.submit(
                get_seq_info, fasta_file,
                bed_file, conda_env, conda_method)
             for bed_file in bed_file_list
             ])
        for task in as_completed(tasks):
            try:
                task.result()
            except Exception as e:
                print(f'An error occurred: {e}')
    print('all flank sequence retrieved       ℹ️ ')

    seq_files_str = ','.join(f'{bed_file}.seq' for bed_file in bed_file_list)

    _cmdln: str
    _cmdln = (f'python {complexity_script} '
              f'-i {seq_files_str} '
              f'-k 70 '
              f'-c 70 '
              f'-p {parallel}')
    _cmdln = (f'conda run -n {conda_env} ' + _cmdln
              if conda_method == 'conda_run'
              else f'source activate {conda_env} && ' + _cmdln)
    run(_cmdln, shell=True)
    print('sequence complexity calculated     ℹ️ ')

    _cmdln_ls: list[str] = [(f'cat {tmp_bed_file}.complex >> '
                             f'{tmp_dir}/{run_name}.seq '
                             f'&& rm {tmp_bed_file}.seq {tmp_bed_file}.complex')
                            for tmp_bed_file in bed_file_list]

    for _cmdln in _cmdln_ls:
        run(_cmdln, shell=True)  # cat cmd has to run by turn.
    run(f'zstd -T{parallel} -19 --rm {tmp_dir}/{run_name}.seq', shell=True)


def parse_bedtools_output(run_name: str, tmp_dir: Path):
    retrieve_flank = 70

    _df: pl.DataFrame
    _df = (pl.scan_csv(source=tmp_dir / f'{run_name}.seq.zst',
                       has_header=False, separator='\t',
                       schema={'chrom': pl.String, 'start': pl.Int64,
                               'end': pl.Int64, 'retrieved_seq': pl.String,
                               'ShannonEntropy_70': pl.Float64,
                               'BWT_ratio_70': pl.Float64})
             .drop_nulls()
             .unique(subset=['chrom', 'start', 'end'])
             .with_columns(pl.col('retrieved_seq').str.to_uppercase())
             .with_columns([pl.col('retrieved_seq')  # Slice the retrieved_seq column to create new columns
                              .str  # for each value in anno_flank_ls, gc_flank_ls and complexity_ls
                              .slice(retrieve_flank - _s,
                                     _s * 2 + 1)
                              .alias(f'seq_{_s}')
                            for _s in (5, 70)])
             .with_columns([pl.col(f'seq_{_s}')
                              .str
                              .count_matches(_b)
                              .cast(pl.Int16)  # the dtype before cast is u32, causing a bug.
                              .alias(f'{_b}_{_s}')
                            for _s in (5, 70)
                            for _b in ('G', 'C')])
             .with_columns((pl.sum_horizontal('G_70', 'C_70') / (70 * 2 + 2))
                           .alias('GC%_70'))
             .with_columns(
                 ((pl.col('G_70') - pl.col('C_70')) / pl.sum_horizontal('G_70',
                                                                        'C_70'))
                 .alias('GC_skew_70'))
             .with_columns([pl.col('retrieved_seq')
                              .str
                              .slice(offset=retrieve_flank + i,
                                     length=1)
                              .alias(j)
                            for i, j in [(-2, 'b2'), (-1, 'b1'),
                                         (0, 'cytosine'),
                                         (1, 'a1'), (2, 'a2')]])
             .with_columns(pl.when(pl.col('cytosine') == 'C')
                             .then(pl.lit('+'))
                             .when(pl.col('cytosine') == 'G')
                             .then(pl.lit('-'))
                             .otherwise(pl.lit('.'))  # . means no strand info
                             .alias('strand'))
             .with_columns(pl.when(pl.col('cytosine') == 'C')
                             .then(pl.when(pl.col('a1') == 'G').then(pl.lit('CpG'))
                                     .when(pl.col('a2') == 'G').then(pl.lit('CHG'))
                                     .otherwise(pl.lit('CHH')))
                             .when(pl.col('cytosine') == 'G')
                             .then(pl.when(pl.col('b1') == 'C').then(pl.lit('CpG'))
                                     .when(pl.col('b2') == 'C').then(pl.lit('CHG'))
                                     .otherwise(pl.lit('CHH')))
                             .otherwise(pl.lit('?'))
                             .alias('type'))
             .with_columns((pl.col('seq_70')
                              .str
                              .count_matches(r'(GC|CG)')
                              .cast(pl.Int16) / pl.sum_horizontal('G_70',
                                                                  'C_70'))
                           .alias('CpG_GC_ratio_70'))
             .with_columns([pl.col('start') + retrieve_flank,  # Add retrieve_flank to the start column and
                            pl.col('end') - retrieve_flank])
             .select(['chrom', 'start', 'end', 'strand', 'type', 'seq_5',
                      'GC%_70', 'GC_skew_70', 'CpG_GC_ratio_70',
                      'ShannonEntropy_70', 'BWT_ratio_70'])
             .drop_nulls()
             .collect())

    _df.write_parquet(tmp_dir / f'{run_name}.seq.parquet.lz4',
                      compression='lz4')
    del _df
    gc.collect()


def retrieve_annotation(run_name: str, bed_file_list: list[str],
                        tmp_dir: Path,
                        parallel: int = 8,
                        docker_image: str = 'annotatr:1.33.0',
                        annotatr_script: Path = ANNOTATR_SCRIPT,
                        singularity: bool = False):
    annotatr_dir = annotatr_script.parent.resolve()
    annotatr_name = annotatr_script.name
    annotatr_tmp = tmp_dir / f'{run_name}.annotatr'
    bed_file_docker_paths: str = (','.join([f'/data/tmp/{Path(bed_file).name}'
                                            for bed_file in bed_file_list]))

    _rscript_cmdln: str
    if singularity:
        _rscript_cmdln = ('singularity exec '
                          '--cleanenv --no-home '
                          f'--cpus {parallel}'
                          f'-B {tmp_dir.resolve()}:/data/tmp:rw '
                          f'-B {annotatr_dir}:/data/annotatr '
                          f'{docker_image} Rscript '
                          f'/data/annotatr/{annotatr_name} '
                          f'--input {bed_file_docker_paths} '
                          f'--cores {parallel}')
    else:
        _rscript_cmdln = ('docker run --rm '
                          f'--cpus {parallel} '
                          f'-v {tmp_dir.resolve()}:/data/tmp '
                          f'-v {annotatr_dir}:/data/annotatr '
                          f'{docker_image} Rscript '
                          f'/data/annotatr/{annotatr_name} '
                          f'--input {bed_file_docker_paths} '
                          f'--cores {parallel}')

    run(_rscript_cmdln, shell=True)

    _cmdln_ls: list[str] = [(f'cat {tmp_bed_file}.annotatr >> {annotatr_tmp}'
                             f' && rm {tmp_bed_file}.annotatr')
                            for tmp_bed_file in bed_file_list]

    for _cmdln in _cmdln_ls:
        run(_cmdln, shell=True)  # rm cmd has to run by turn.
    run(f'zstd -T{parallel} -19 --rm {annotatr_tmp}', shell=True)


def merge_anno_info(run_name: str, tmp_dir: Path):
    _schema = {'chrom': pl.String, 'start': pl.Int64,
               'end': pl.Int64, 'gene_symbol': pl.String,
               'anno': pl.String}
    _base_cols = ['chrom', 'start', 'end']

    annotations: list[AnnotationMap] = [promoter_mapping,
                                        enhancer_mapping,
                                        lncrna_mapping,
                                        cpg_mapping,
                                        location_mapping]

    anno_df: pl.DataFrame
    anno_df = (pl.scan_csv(tmp_dir / f'{run_name}.annotatr.zst',
                           separator='\t', schema=_schema)
                 .with_columns(pl.col('anno')
                                 .replace_strict(ANNO_REPLACE_DICT,
                                                 default='intergenic',
                                                 return_dtype=pl.String))
                 .group_by(_base_cols)
                 .agg(pl.col('anno').unique().alias('anno_list'),
                      pl.col('gene_symbol').unique().alias('gene_symbols'))
                 .with_columns([pl.col('anno_list')
                                  .list
                                  .eval(pl.element()
                                          .replace_strict(_AnnoMap.anno2order,
                                                          default=_AnnoMap.default_order,
                                                          return_dtype=pl.Int64))
                                  .list.min()
                                  .replace_strict(_AnnoMap.order2anno,
                                                  default=_AnnoMap.default_anno,
                                                  return_dtype=_AnnoMap.dtype)
                                  .alias(_AnnoMap.name)
                                for _AnnoMap in annotations])
                 .with_columns(pl.col('gene_symbols').list.join(separator=','))
                 .with_columns(pl.col('start') - 1)
                 .select(_base_cols + [
                     'gene_symbols'
                 ] + [_AnnoMap.name for _AnnoMap in annotations]).collect())
    anno_df.write_parquet(tmp_dir / f'{run_name}.annotatr.parquet.lz4',
                          compression='lz4')
    del anno_df
    gc.collect()


def check_parquet_sliced(run_name: str, tmp_dir: Path, flank: bool) -> tuple[bool, list[str]]:
    sliced = False
    tmp_files: list[str] = []
    if flank:
        tmp_file_pkl: Path = tmp_dir / f'{run_name}.flank_list.pkl'
        task_id = '1.1'
    else:
        tmp_file_pkl = tmp_dir / f'{run_name}.bed_list.pkl'
        task_id = '2.1'

    if tmp_file_pkl.exists():
        with open(tmp_file_pkl, 'rb') as f:
            tmp_files = pickle_load(f)
            if all(Path(tmp_file).exists() for tmp_file in tmp_files):
                logger.info(f'✅ Task {task_id} Using existing {len(tmp_files)} slices')
                sliced = True
            else:
                logger.info(f'❗Task {task_id} Some slices are missing, removing all slices')
                for tmp_file in tmp_files:
                    Path(tmp_file).unlink(missing_ok=True)

    return sliced, tmp_files


def annotate(run_name: str,
             parquet_df: pl.DataFrame,
             tmp_dir: Path,
             output_dir: Path,
             parallel: int):
    final_df: pl.DataFrame
    final_df = (parquet_df.lazy()
                          .join(other=pl.scan_parquet(tmp_dir / f'{run_name}.seq.parquet.lz4')
                                        .unique(keep='first'),
                                on=['chrom', 'start', 'end'], how='left', validate='1:1')
                          .join(other=pl.scan_parquet(tmp_dir / f'{run_name}.annotatr.parquet.lz4')
                                        .unique(keep='first'),
                                on=['chrom', 'start', 'end'], how='left', validate='1:1')
                          .collect())
    final_df.write_parquet(output_dir / f'{run_name}.parquet.lz4', compression='lz4')

    zstd_compressor = ZstdCompressor(level=19, threads=parallel)
    with zstd_compressor.stream_writer(writer=open(file=output_dir / f'{run_name}.anno.bedgraph.zst',
                                                   mode='wb+')) as _writer:
        final_df.write_csv(_writer, include_header=True, separator='\t', line_terminator='\n')
    Path(output_dir / f'{run_name}.parquet.lz4').chmod(0o777)
    Path(output_dir / f'{run_name}.anno.bedgraph.zst').chmod(0o777)


def main():
    arg_parser: ArgumentParser = ArgumentParser(description=DESCRIPTION_MESSAGE,
                                                formatter_class=RawTextHelpFormatter)
    # elaborate all params in (dummy-proof) detail to clarify the usage of this script.
    arg_parser.add_argument('-i', '--input', type=Path, required=True,
                            help='the file path of bedgraph in parquet format to annotate. '
                                 'If a directory is given, all parquet files in the directory '
                                 'will be merged and processed.')
    arg_parser.add_argument('--whitelist', type=str, default=None,
                            help='If provided and input is a directory, only files with names '
                                 'containing any string in whitelist will be processed. '
                                 'E.g. "D5,BS1" will only process files with "D5" or "BS1" '
                                 'in their names. Strings should be separated by commas.')
    arg_parser.add_argument('-n', '--name', type=str, default=None,
                            help='the file name of the output annotated bedgraph in parquet format. '
                                 'If input is a parquet file, and name is not provided, the '
                                 'output file will be named as the input file. '
                                 'If input is a directory, and name is not provided, '
                                 'the output file will be named as union.parquet.lz4')
    arg_parser.add_argument('-o', '--output-dir', dest='output_dir',
                            type=Path, required=True,
                            help='the dir to output the annotated bedgraph')
    arg_parser.add_argument('-t', '--tmp-dir', dest='tmp_dir',
                            type=Path, required=True,
                            help='the dir to store temporary bed file')
    arg_parser.add_argument('-r', '--ref', type=Path, required=True,
                            help='the reference fasta file to retrieve sequence. '
                                 'In this script, automatic reference selection is not supported.')
    arg_parser.add_argument('-p', '--parallel', default=1,
                            type=int, help='the threads to use for parallel processing')
    arg_parser.add_argument('-e', '--conda-env', dest='conda_env',
                            type=validate_conda_env, required=True,
                            help='the name or dir path of the conda env containing seqkit')
    arg_parser.add_argument('-m', '--conda_method', type=str,
                            choices=['conda_run', 'source_activate'],
                            default='conda_run',
                            help='either "conda_run"  or "source_activate"')
    arg_parser.add_argument('--annotatr-image', dest='annotatr_image',
                            type=validate_docker_image, required=True,
                            help='docker image containing annotatr package')
    arg_parser.add_argument('-s', '--annotatr-script', dest='annotatr_script',
                            default=ANNOTATR_SCRIPT, type=Path,
                            help='path of R script to run annotatr')
    arg_parser.add_argument('-x', '--complexity-script', dest='complexity_script',
                            default=SEQ_COMPLEX_SCRIPT, type=Path,
                            help='path of python script to calculate sequence complexity')
    arg_parser.add_argument('--cutoff', dest='cutoff',
                            type=int, default=5,
                            help='the depth cutoff to filter the input bedgraph. '
                                 'Only regions with depth >= depth_cutoff will be kept. '
                                 'Default is 5.')

    args: Namespace = arg_parser.parse_args()

    input_path: Path = args.input
    output_dir: Path = args.output_dir
    tmp_dir: Path = args.tmp_dir
    ref_fasta: Path = args.ref
    ref_fasta_fai: Path = Path(f'{args.ref}.fai')
    conda_env: str = args.conda_env
    conda_method: Literal['conda_run', 'source_activate'] = args.conda_method
    annotatr_image: str = args.annotatr_image
    annotatr_script: Path = args.annotatr_script
    complexity_script: Path = args.complexity_script
    depth_cutoff: int = args.cutoff
    cores: int = args.parallel

    parquet_files: list[Path]
    run_name: str

    if not input_path.exists():
        raise FileNotFoundError(f'{input_path} does not exist')

    if input_path.is_dir():
        parquet_files = list(input_path.glob('*.parquet.lz4'))
        if args.whitelist:
            whitelist: list[str] = args.whitelist.split(',')
            parquet_files = [f for f in parquet_files if any(w in f.name for w in whitelist)]
        if not parquet_files:
            raise ValueError(f'No parquet files found in {input_path}')
        if args.name is None:
            run_name = 'union'
        else:
            run_name = args.name
    else:
        parquet_files = [input_path]
        if args.name is None:
            run_name = input_path.stem
        else:
            run_name = args.name

    for _n, _f in {'Reference fasta file': ref_fasta,
                   'Reference fasta index file': ref_fasta_fai,
                   'Annotatr script': annotatr_script,
                   'Complexity script': complexity_script}.items():
        if not _f.exists():
            raise FileNotFoundError(f'{_n}: {_f} is not a valid file path')

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Running Config:\n'
                '==============================\n'
                f'Run Name: {run_name}\n'
                f'Input Path: {parquet_files}\n'
                f'Output Directory: {output_dir}\n'
                f'Temporary Directory: {tmp_dir}\n'
                f'Reference Fasta: {ref_fasta}\n'
                f'Parallel Threads: {cores}\n'
                f'Conda Environment for retrieving sequences: {conda_env}\n'
                f'Conda Method: {conda_method}\n'
                f'Docker Image for annotatr: {annotatr_image}\n'
                f'Annotatr Script: {annotatr_script}\n'
                f'Complexity Script: {complexity_script}\n'
                f'Depth Cutoff: {depth_cutoff}\n'
                '==============================\n')

    parquet_df: pl.DataFrame
    if len(parquet_files) == 1:
        parquet_df = (pl.scan_parquet(parquet_files[0])
                        .filter(pl.col('chrom').is_in(CHROMOSOMES),
                                pl.col('depth') >= depth_cutoff)
                        .unique(subset=['chrom', 'start', 'end'],
                                keep='first')
                        .select('chrom', 'start', 'end')
                        .collect())
    else:
        parquet_df = (pl.concat([pl.scan_parquet(f).select('chrom', 'start', 'end', 'depth')
                                 for f in parquet_files])
                        .filter(pl.col('chrom').is_in(CHROMOSOMES),
                                pl.col('depth') >= depth_cutoff)
                        .unique(subset=['chrom', 'start', 'end'],
                                keep='first')
                        .select('chrom', 'start', 'end')
                        .collect())

    if parquet_df.is_empty():
        raise ValueError('No valid regions found in the input bedgraph files after filtering.')

    logger.info(f'✅ Parquet dataframe read, containing {parquet_df.shape[0]} cytosines.')
    logger.info('ℹ️ Task 1   Retrieving sequence and calculating complexity')

    seq_retrieved: bool = Path(tmp_dir / f'{run_name}.seq.zst').is_file()
    seq_parsed: bool = Path(tmp_dir / f'{run_name}.seq.parquet.lz4').is_file()

    if not seq_parsed:
        if not seq_retrieved:
            (flank_sliced,
             tmp_flank_files) = check_parquet_sliced(run_name=run_name, tmp_dir=tmp_dir, flank=True)
            if not flank_sliced:
                logger.info('ℹ️ Task 1.1 Splitting parquet dataframe into slices')
                # Generate temporary bed files from the parquet dataframe
                tmp_flank_files = generate_bed(df=parquet_df,
                                               tmp_dir=tmp_dir,
                                               retrieve_flank=70,
                                               flank=True)
                with open(tmp_dir / f'{run_name}.flank_list.pkl', 'wb+') as f:
                    pickle_dump(tmp_flank_files, f)
                logger.info(f'✅ Task 1.1 parquet dataframe divided into {len(tmp_flank_files)} slices finished')

            logger.info('ℹ️ Task 1.2 Retrieving sequence')
            retrieve_sequence(run_name=run_name, bed_file_list=tmp_flank_files,
                              fasta_file=ref_fasta,
                              tmp_dir=tmp_dir.as_posix(), conda_env=conda_env,
                              parallel=cores, conda_method=conda_method,
                              complexity_script=complexity_script)
            logger.info(f'✅ Task 1.2 Sequence retrieved, saved at {tmp_dir}/{run_name}.seq.zst')

        parse_bedtools_output(run_name=run_name, tmp_dir=tmp_dir)
        logger.info(f'✅ Task 1   Sequence parsed and saved  at {tmp_dir}/{run_name}.seq.parquet.lz4')
    else:
        logger.info('✅ Task 1   Sequence parsed')

    annotatr_finished: bool = (tmp_dir / f'{run_name}.annotatr.parquet.lz4').is_file()
    annotatr_retrieved: bool = (tmp_dir / f'{run_name}.annotatr.zst').is_file()

    logger.info('ℹ️ Task 2   Retrieving annotation from annotatr database')
    if not annotatr_finished:
        if not annotatr_retrieved:
            (bed_sliced,
             tmp_bed_files) = check_parquet_sliced(run_name=run_name, tmp_dir=tmp_dir, flank=False)
            if not bed_sliced:
                logger.info('ℹ️ Task 2.1 Splitting parquet dataframe into slices')
                # Generate temporary bed files from the parquet dataframe
                tmp_bed_files = generate_bed(df=parquet_df,
                                             tmp_dir=tmp_dir,
                                             flank=False)
                with open(tmp_dir / f'{run_name}.bed_list.pkl', 'wb+') as f:
                    pickle_dump(tmp_bed_files, f)
                logger.info(f'✅ Task 2.1 parquet dataframe divided into {len(tmp_bed_files)} slices finished')
            logger.info('ℹ️ Task 2.2 Retrieving annotation')
            retrieve_annotation(run_name=run_name, bed_file_list=tmp_bed_files,
                                tmp_dir=tmp_dir, parallel=cores,
                                docker_image=annotatr_image,
                                annotatr_script=annotatr_script)
            logger.info(f'✅ Task 2.2 Annotation retrieved, saved at {tmp_dir}/{run_name}.annotatr.zst')
        merge_anno_info(run_name=run_name, tmp_dir=tmp_dir)
        logger.info(f'✅ Task 2   Annotation merged, saved at {tmp_dir}/{run_name}.annotatr.parquet.lz4')

    logger.info('ℹ️ Task 3   Merging all info')

    annotate(run_name=run_name,
             parquet_df=parquet_df,
             tmp_dir=tmp_dir,
             output_dir=output_dir,
             parallel=cores)
    logger.info('✅ Task 3   Annotation finished, saved at\n'
                f'1: {output_dir}/{run_name}.parquet.lz4\n'
                f'2: {output_dir}/{run_name}.anno.bedgraph.zst\n')


if __name__ == '__main__':
    main()
