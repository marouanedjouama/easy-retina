import os
import cv2
import insightface
import pickle as pkl
import numpy as np
import faiss
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

data_output_path = os.path.join(ROOT, "data.pkl")

model_path = os.path.join(ROOT,"models/resnet_34.onnx")
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"❌ File not found: {model_path}")

faces_dir = os.path.join(ROOT.parents[0], "face_dataset")
if not os.path.isdir(faces_dir):
    raise FileNotFoundError(f"❌ Directory not found: {faces_dir}")


vec_dataset = {}

recognition_model = insightface.model_zoo.get_model(model_path)

for dirname in os.listdir(faces_dir):
    dirpath = os.path.join(faces_dir, dirname)
    embeds = []

    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        image = cv2.imread(filepath)
        if image is None:
            print(f"⚠️ Could not load image: {filepath}")
            continue  # skip this file

        embd = recognition_model.get_feat(image)
        embeds.append(embd)

    vec_dataset[dirname] = embeds
    print(f"Processed {dirname}")


clustered_data = {}
# cluster embeddings by averaging them
for person, embeds in vec_dataset.items():
    person_embeddings = np.array(embeds)
    person_embeddings = np.squeeze(person_embeddings, axis=1)  # shape (N, D)
    mean_emb = np.mean(person_embeddings, axis=0)
    mean_emb /= np.linalg.norm(mean_emb)
    clustered_data[person] = mean_emb


d = recognition_model.output_shape[-1]   # dimension
index = faiss.IndexFlatL2(d)   # build the index
vectors = np.array(list(clustered_data.values()))
index.add(vectors)


data = {"labels": list(clustered_data.keys()), "index": index}

with open(data_output_path, "wb") as f:
    pkl.dump(data, f)

print(f"Data saved to {data_output_path}")
