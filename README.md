# MethCali

A workflow to calibrate the detected beta values from NGS-based methylation sequencing data. It's designed to be used right after the `dna_methylation_smk` workflow, which generate the methylation bedgraph file from our recommended `trim-galore`+`bwa-meth`+`gatk MarkDuplicates`+`asTair` for WGBS/EM-seq, `fastp`+`lighter`+`gem3`+`gencore`+`rastair` for TAPS, and  `trim-galore`+`bwa-meth`+`asTair` for RRBS.

Specially, the cytosines with `depth >= 10` will be calibrated using `Autogluon`, while the cytosines with depth calibrated using `XGBoost`. Cytosines on chrM/lambda/pUC19 will keep original.

> [!TIP]
> Before run this workflow, please use `dna_methylation_smk` workflow to generate the methylation bedgraph file as above, and use the `dna_methylation_smk/utils/count2eqa.py` to generate the formatted bedgraph in parquet.lz4 file as the input of this workflow.
> For multiple samples, please save or link the parquet.lz4 files in single directory.

This workflow contains following steps:

## Cytosine-level annotation

all the cytosines with depth >=3 will be annotated with biological features, including `seq_5`, `GC_skew_70`, `CpG_GC_ratio_70`, `ShannonEntropy_70`, `BWT_ratio_70`, `cpg`, `location`, `promoter` and `enhancer`.

```bash
python "${methcali_dir}/src/cytosine_annotation/annotate.py" \
  -i "${bedgraph_dir}" \
  --whitelist D5 -n D5 \
  -o "${data_dir}/union" \
  -t "${data_dir}/tmp" \
  -r "${ref_dir}"/GRCh38.fa \
  -p 12 \
  -e genomic_tools \
  -m conda_run \
  --annotatr-image annotatr:1.33.0 \
  -s src/cytosine_annotation/annotatr_bed_parallel.R \
  -x src/cytosine_annotation/seq_complex_parallel.py \
  --cutoff 3
```

for example:
```bash
for w n r in D5,D6,T1,T2,T3,T4 D5 D5 F7 F7 F7 M8 M8 M8 BC BC BC BL BL BL; do
  echo -e "${w}\t${n}\t${r}"
  python src/cytosine_annotation/annotate.py \
    -i /opt/formatted \
    --whitelist $w -n $n \
    -o /opt/union \
    -t /opt/tmp \
    -r /hot_warm_data/ref/quartet/DNA/custom_genome/${r}.fa \
    -p 128 \
    -e genomic_tools \
    -m conda_run \
    --annotatr-image annotatr:1.33.0 \
    -s src/cytosine_annotation/annotatr_bed_parallel.R \
    -x src/cytosine_annotation/seq_complex_parallel.py \
    --cutoff 5
done
```

### Preparation

It requires:

   + reference fasta file (GRCh37/GRCh38/T2T-CHM13), same as the one used in `dna_methylation_smk` workflow. In our study, we use GRCh38.p14 as the reference, which you could download from NCBI/UCSC/other sources.

   + annotatr docker image described in `envs/annotatr.Dockerfile`. Run the following commands to build it:

```bash
cd envs
docker built \
  -t annotatr:methcali \
  -f annotatr.Dockerfile \
  --build-arg CORES=${cores}
```

You should provide cores >= 8 to accelerate the installation of R packages.

   + polars conda env described in `envs/polars.yaml`. Run the following commands to build it:

```bash
mamba create -n polars -f envs/polars.yaml
```

   + compile conda env described in `envs/compile.yaml` to generate `shannon.so` from `shannon.c`.

```bash
gcc -O3 -Wall -shared -std=c99 \
  -fPIC $(python3 -m pybind11 --includes) \
  src/cytosine_annotation/shannon.c -o src/cytosine_annotation/shannon.so \
  -I${miniforge_dir}/pkgs/python-3.12.11-*_cpython/include/python3.12
```

The directory containing .h head files for -I param can be found in miniforge dir.

## Sample-level annotation

It only requires the formatted `parquet.lz4`, and -m param to specify the sequencing method used for this sample.
Only `BS`, `EM`, `PS` and `RR` are supported for now.

```bash
for sample in $(find ../best_pipeline -name '*parquet.lz4' -type f | sort); do
python "${methcali_dir}/src/sample_annotation/distribution.py" stat \
  -i ${sample} \
  -m BS \
  -o "${metadata_dir}"
done
```

for example:
```bash
for prefix in BS EM PS RR; do
  find /opt/formatted -name "${prefix}*parquet.lz4" -type f | sort |\
    parallel -j 48 --bar \
      python src/sample_annotation/distribution.py stat \
        -i {} -m "$prefix" \
        -o /opt/data/distribution
done
```

There will be a lab_sample_rep-metadata.pkl in the metadata directory.

(Not necessary) You can merge all the metadata files into a single parquet file for statistic usage:
```bash
python "${methcali_dir}/src/sample_annotation/distribution.py" merge \
  -i "${metadata_dir}" \
  -o "${metadata_dir}/merged_metadata.parquet.lz4"
```

## Division of low- and high- depth cytosines

| contigs                | depth range | goto                                          |
|:---------------------- | :---------- | :-------------------------------------------- |
| chr1-chr22, chrX, chrY | `>= 10 `    | annotated; calibrated by Autogluon + LightGBM |
| chr1-chr22, chrX, chrY | ` < 10`     | annotated; calibrated by JIT XGboost          |
| chrM, lambda, pUC19    | `any`       | kept original                                 |

```bash
python "${methcali_dir}/src/division/divide.py" \
  -i "${bedgraph_dir}/${sample}.parquet.lz4" \
  --annotation "${data_dir}/union/sample.parquet.lz4" \
  --metadata "${metadata_dir}/${sample}.pkl" \
  -co "${data_dir}/controls/${sample}.parquet.lz4" \
  -lo "${data_dir}/low_depth/${sample}.parquet.lz4" \
  -ho "${data_dir}/high_depth/${sample}.parquet.lz4" \
  --cutoff 10
```

for example:
```bash
for label in D5 D6 F7 M8 T1 T2 T3 T4 BC BL; do
  find /opt/formatted -type f -name "*.parquet.lz4" |\
  cut -d / -f8 | sed 's/\.parquet\.lz4$//' | sort |\
    parallel -j 16 --bar \
      "python src/division/divide.py \
        -i /opt/formatted/{}.parquet.lz4 \
        -a /opt/union/${label}.parquet.lz4 \
        -m /opt/data/distribution/{}-metadata.pkl \
        -co /opt/data/control/{}.parquet.lz4 \
        -lo /opt/data/low_depth/{}.parquet.lz4 \
        -ho /opt/data/high_depth/{}.parquet.lz4 \
        -c 10"
done
```


## Calibration

Use LightGBM model in Autogluon TabularPredictor to calibrate the beta values of high-depth cytosines.

```bash
python "${methcali_dir}/src/calibration/autogluon.py" \
  -i "${data_dir}/high_depth/${sample}.parquet.lz4" \
  -o "${data_dir}/calibrated_high_depth/${sample}.parquet.lz4" \
  -md "${methcali_dir}/model/autogluon_model" \
  -mn LightGBMXT_BAG_L1
```

For exmaple:
```bash
# L2 taks 33m 18s
time python src/calibration/autogluon.py \
  -i /opt/data/high_depth/BS3_D6_2.parquet.lz4 \
  -o /opt/data/calibrated_high_depth/BS3_D6_2.parquet.lz4 \
  -md models/2025-07-18-14-49 \
  -mn LightGBMXT_BAG_L2 && echo -e "2025-07-18-14-49\tLightGBMXT_BAG_L2"
```

You could choose the `2025-07-18-14-49` full model or `best_cloned` post-processed model

### Preparation

It requires:

   + the pre-trained model directory `across_methods` in `autogluon_models.tar.zst` downloaded and extracted.

   + conda env for Autogluon, described in its [offical documentation](https://auto.gluon.ai/stable/install.html).
      There is an example below:

     for CPU:

```bash
mamba create -n ag_cpu -c conda-forge python=3.10
mamba activate ag_cpu
pip install "autogluon.tabular[all,imodels,skex,skl2onnx]" "polars[pandas,pyarrow]" \
  --resume-retries 6 --extra-index-url https://download.pytorch.org/whl/cpu
```

​     for GPU:

```bash
mamba create -n ag_gpu -c conda-forge python=3.11
mamba activate ag_gpu
pip install "lightgbm[arrow,pandas]" \
  --no-binary lightgbm \
  --config-settings=cmake.define.USE_GPU=ON
pip install "autogluon.tabular[all,imodels,skex,skl2onnx]" "polars[pandas,pyarrow]" \
  --extra-index-url https://download.pytorch.org/whl/cu128
 ```

​     If you are using CUDA 12.9, use `cu129` instead of `cu128` in the above command.
​     For GPU env, don't forget to use the following [python function](https://github.com/microsoft/LightGBM/issues/3939) to check if GPU support is on:

```python
import lightgbm
import numpy as np
def check_gpu_support():
    try:
        data = np.random.rand(50, 2)
        label = np.random.randint(2, size=50)
        train_data = lightgbm.Dataset(data, label=label)
        params = {'num_iterations': 1, 'device': 'gpu'}
        gbm = lightgbm.train(params, train_set=train_data)
        return True
    except Exception as e:
        return False
check_gpu_support()
```

If it returns `True`, then you are good to go. If the warning about LightGBM without GPU support still appears, please try autolguon offical docker image instead.


## Exploration
Use XGBoost to train a small sample-specific quantile regression model on calibrated high-depth cytosines, and then impute the beta values of low-depth cytosines.
```bash
python "${methcali_dir}/src/imputation/xgboost_quantile.py" \
-hi "${data_dir}/calibrated_high_depth/${sample}.parquet.lz4" \
-mi "${data_dir}/low_depth/${sample}.parquet.lz4" \
-o "${data_dir}/calibrated_low_depth/${sample}" \
-c 24 -g 1 -n 50
```

## Merge
```bash
python "${methcali_dir}/src/division/merge.py" \
  -oi "${data_dir}/all_formatted/${sample}.parquet.lz4" \
  -ci "${data_dir}/data/controls/${sample}.parquet.lz4" \
  -li "${data_dir}/data/calibrated_low_depth/${sample}.parquet.lz4" \
  -hi "${data_dir}/data/calibrated_high_depth/${sample}.parquet.lz4" \
  -o "${data_dir}/data/merged/${sample}.parquet.lz4"
```
