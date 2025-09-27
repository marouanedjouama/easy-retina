import os
import cv2
import insightface
import pickle as pkl
import numpy as np
from pathlib import Path
import argparse


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

data_output_path = os.path.join(ROOT,"data.pkl")
if not os.path.isfile(data_output_path):
    raise FileNotFoundError(f"❌ File not found: {data_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Face recognition inference")
    parser.add_argument("--image", required=True, type=str, help="Path to the test image")
    parser.add_argument("--model",type=str, default="models/resnet_34.onnx", help="Path to the recognition model image")
  
    return parser.parse_args()


def main(args):
    test_image_path = args.image
    if not os.path.isfile(test_image_path):
        raise FileNotFoundError(f"❌ File not found: {test_image_path}")

    image = cv2.imread(test_image_path)
    if image is None:
        print(f"⚠️ Could not load image: {test_image_path}")
        exit(1)

    model_path = args.model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"❌ File not found: {model_path}")


    # load data
    with open("data.pkl", "rb") as f:
        data = pkl.load(f)

    labels = data["labels"]
    index = data["index"]

    # load model and get embedding
    recognition_model = insightface.model_zoo.get_model(model_path, providers=['CPUExecutionProvider'])
    embd = recognition_model.get_feat(image)
    embd = embd / np.linalg.norm(embd)

    k = 1
    D, I = index.search(embd, k)
    prediction = labels[I[0][0]]

    print(f"Prediction: {prediction} (distance: {D[0][0]:.2f})")

if __name__ == "__main__":
    args = parse_args()
    main(args)