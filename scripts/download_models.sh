#!/bin/bash
# WebFace600K recognition model https://github.com/deepinsight/insightface/tree/master/model_zoo
# Download command: bash ./scripts/download_models.sh

set -euo pipefail

# Directories
DET_MODEL_DIR="detection/models"
REC_MODEL_DIR="recognition/models"

mkdir -p "$DET_MODEL_DIR" "$REC_MODEL_DIR"

# Model IDs
declare -A MODELS=(
  ["$REC_MODEL_DIR/WebFace600K.onnx"]="1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg"
  ["$REC_MODEL_DIR/resnet34.onnx"]="1G1oeLkp_b3JA_z4wGs62RdLpg-u_Ov2Y"
  ["$DET_MODEL_DIR/scrfd_2.5g.onnx"]="1f6T5LzpGroJwF5zZr-FsTf6Jv9ZmxUEV"
)

# Download function
download_model() {
  local output="$1"
  local file_id="$2"
  if [[ -f "$output" ]]; then
    echo "[SKIP] $output already exists."
  else
    echo "[DOWNLOAD] $output"
    gdown "$file_id" -O "$output"
  fi
}

# Loop through models
for output in "${!MODELS[@]}"; do
  download_model "$output" "${MODELS[$output]}"
done
