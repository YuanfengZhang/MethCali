labs=(
    "BS1" "BS2" "BS3" "BS4"
    "EM1" "EM2" "EM3" "EM4"
    "PS1" "PS2" "PS3" "RR1"
)

labels=(
    "D5" "D6" "F7" "M8"
    "T1" "T2" "T3" "T4"
    "BC" "BL"
)

reps=(
    "1" "2"
)

high_depth_dir="/mnt/eqa/zhangyuanfeng/methylation/best_pipeline/data/high_depth"
calibrated_dir="/mnt/eqa/zhangyuanfeng/methylation/best_pipeline/data/calibrated_high_depth"

for lab in "${labs[@]}"; do
  for label in "${labels[@]}"; do
    for rep in "${reps[@]}"; do
      if [ ! -f "${calibrated_dir}/${lab}_${label}_${rep}.parquet.lz4" ]; then
        python src/calibration/autogluon.py \
          -i "${high_depth_dir}/${lab}_${label}_${rep}.parquet.lz4" \
          -o "${calibrated_dir}/${lab}_${label}_${rep}.parquet.lz4" \
          -md /hot_warm_data/zhangyuanfeng/methylation/models/2025-07-18-14-49 \
          -mn LightGBMXT_BAG_L2
      fi
    done
  done
done