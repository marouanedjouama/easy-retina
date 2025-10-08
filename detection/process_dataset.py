"""
this script processes full images and returns cropped faces found (works with data structured according the readme file) 
"""

import os
import cv2
from pathlib import Path
from detect import SCRFD
import random
import shutil


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

faces_dir = os.path.join(ROOT.parents[0], "face_dataset")
if not os.path.isdir(faces_dir):
    raise FileNotFoundError(f"❌ Directory not found: {faces_dir}")


new_faces_dir = os.path.join(ROOT.parents[0], "face_crops_dataset")
os.makedirs(new_faces_dir, exist_ok=True)

recognition_test_dir = os.path.join(ROOT.parents[0], "recognition_test")
os.makedirs(recognition_test_dir, exist_ok=True)


print(ROOT)
print(faces_dir)
print(new_faces_dir)

exit()


model_path = os.path.join(ROOT,"models/scrfd_2.5g.onnx")
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"❌ File not found: {model_path}")


detector = SCRFD(model_path=model_path, conf_thres=0.4)
pad = 5  # pixels to pad around the face bounding box
for dirname in os.listdir(faces_dir):

    dirpath = os.path.join(faces_dir, dirname)

    target_dir_path = os.path.join(new_faces_dir, dirname)
    os.makedirs(target_dir_path, exist_ok=True)

    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)

        image = cv2.imread(filepath)
        if image is None:
            print(f"⚠️ Could not load image: {filepath}")
            continue  # skip this file

        boxes_list, _ = detector.detect(image)

        h, w = image.shape[:2]  # image height and width
        for i, box in enumerate(boxes_list):
            x1, y1, x2, y2, _ = [int(v) for v in box]  # make sure they're integers

             # Clip to image boundaries, coor can be outside of image shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                print(f"Skipping face {i}, invalid box: {x1,y1,x2,y2}")
                continue

            
            face = image[y1:y2, x1:x2]              # crop the face region
            cv2.imwrite(os.path.join(target_dir_path, f'{filename}_face_{i}.jpg'), face)      # save each face

    print(f"Processed {dirname}")



# list all person folders inside new_faces_dir
person_folders = [f for f in os.listdir(new_faces_dir) if os.path.isdir(os.path.join(new_faces_dir, f))]

if len(person_folders) < 2:
    num_to_pick = 1
else:
    num_to_pick = 2


# pick 2 random distinct persons
selected_persons = random.sample(person_folders, num_to_pick)

for person in selected_persons:

    person_path = os.path.join(new_faces_dir, person)
    images = [f for f in os.listdir(person_path) if os.path.isfile(os.path.join(person_path, f))]

    if not images:
        print(f"⚠️ No images found for {person}, skipping...")
        continue

    # pick a random image
    chosen_img = random.choice(images)
    src = os.path.join(person_path, chosen_img)
    dst = os.path.join(recognition_test_dir, f"{person}_{chosen_img}")
    shutil.move(src, dst)
    