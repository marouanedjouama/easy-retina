#!/bin/bash
# WebFace600K recognition model https://github.com/deepinsight/insightface/tree/master/model_zoo
# Download command: bash ./scripts/get_model.sh

MODEL_DIR="models"
mkdir -p $MODEL_DIR

MODEL_PATH="$MODEL_DIR/WebFace600K.onnx"
FILE_ID="1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg"

URL="https://drive.google.com/uc?id=$FILE_ID"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading WebFace600K model..."
    gdown "$URL" -O "$MODEL_PATH"
else
    echo "Model already exists at $MODEL_PATH"
fi