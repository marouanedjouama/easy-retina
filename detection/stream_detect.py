import os
import cv2
import time
from pathlib import Path
from detection.detect import SCRFD
from detection.anti_spoof.FasNat import Fasnet
from detection.helpers import draw_detections
import numpy as np


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

anti_spoof = True

model_path = os.path.join(ROOT, "detection", "models", "scrfd_2.5g.onnx")
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"❌ File not found: {model_path}")


detector = SCRFD(model_path=model_path)
if anti_spoof:
    antispoof_model = Fasnet()

cap = cv2.VideoCapture(0)

frame_count = 0
boxes_list = []

# detect every N frames
N = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % N == 0:  
        boxes_list, _ = detector.detect(frame)


    if anti_spoof:
        results = []
        for box in boxes_list:
            x, y, w, h, _ = box
            is_real, spoof_score = antispoof_model.analyze(img=frame, facial_area=(x, y, w, h))
            results.append((box, is_real, spoof_score))

    else:
        results = boxes_list


    if len(results):
        frame = draw_detections(frame, results, conf_threshold=0.5)

    cv2.imshow("Face Detection Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()