import os
import random
import cv2
import insightface
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

data_output_path = os.path.join(ROOT,"data.pkl")
if not os.path.isfile(data_output_path):
    print(f"❌ File not found: {data_output_path}")

model_path = os.path.join(ROOT,"models/WebFace600K.onnx")
if not os.path.isfile(model_path):
    print(f"❌ File not found: {model_path}")

test_image_path = os.path.join(ROOT, "test_image")
if not os.path.isfile(test_image_path):
    print(f"❌ File not found: {test_image_path}")


recognition_model = insightface.model_zoo.get_model(model_path)

with open("data.pkl", "rb") as f:
    data = pkl.load(f)

labels = data["labels"]
index = data["index"]


image = cv2.imread(test_image_path)
if image is None:
    print(f"⚠️ Could not load image: {test_image_path}")
    exit(1)

embd = recognition_model.get_feat(image)
embd = embd / np.linalg.norm(embd)

k = 1
D, I = index.search(embd, k)
prediction = labels[I[0][0]]

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis("off")
plt.title(f"Prediction: {prediction} (distance: {D[0][0]:.2f})", fontsize=12)
plt.savefig(os.path.join(ROOT, "result.png"))