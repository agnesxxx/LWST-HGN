#!/usr/bin/env bash
set -euo pipefail

# Reproduce the released best AirQuality/C6H6_GT setting.
# Usage:
#   bash run_best.sh [GPU_ID] [DATA_PATH]
# Example:
#   bash run_best.sh 0 data/AirQualityUCI.csv

GPU_ID="${1:-0}"
DATA_PATH="${2:-data/AirQualityUCI.csv}"

python -u train_airquality_c6h6.py \
  --data_path "${DATA_PATH}" \
  --out_dir outputs \
  --target C6H6_GT \
  --input_mode strict \
  --split_mode segment \
  --window 3 \
  --window_list 3 \
  --target_transform log1p \
  --use_corr_filter 1 \
  --corr_remove_n 3 \
  --corr_remove_list 3 \
  --hypergraph_mode static \
  --auto_k_search 1 \
  --k_candidates 3 \
  --fixed_k 3 \
  --use_multiscale_gtc 1 \
  --multiscale_kernel_sizes 3,5,7 \
  --multiscale_dilations 1 \
  --multiscale_fusion gated \
  --use_wavelet_enhance 1 \
  --wavelet_type db2 \
  --wavelet_alpha 0.055 \
  --wavelet_position post_mixer \
  --wavelet_learnable 1 \
  --wavelet_learnable_alpha 0 \
  --wavelet_filter_norm 1 \
  --wavelet_learnable_lowpass 1 \
  --wavelet_lowpass_mode residual \
  --wavelet_lowpass_alpha_ratio 0.5 \
  --wavelet_energy_lambda 1e-4 \
  --loss huber \
  --huber_delta 1.0 \
  --selection_metric nrmse \
  --lr 0.0015 \
  --weight_decay 0.0001 \
  --batch_size 128 \
  --max_epochs 150 \
  --early_stopping_patience 25 \
  --dropout 0.10 \
  --seed 3407 \
  --gpu "${GPU_ID}" \
  --save_predictions 1 \
  --save_adjacency 0 \
  --save_model 0 \
  --verbose 1
