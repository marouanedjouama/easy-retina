import os
import cv2
import time
from pathlib import Path
from detection.detect import SCRFD
from detection.helpers import draw_detections


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

model_path = os.path.join(ROOT, "detection", "models", "scrfd_2.5g.onnx")
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"❌ File not found: {model_path}")

detector = SCRFD(model_path=model_path)

cap = cv2.VideoCapture(0)

frame_count = 0
boxes_list = None

# detect every N frames
N = 5   

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % N == 0:  
        boxes_list, _ = detector.detect(frame)

    if boxes_list is not None:
        frame = draw_detections(frame, boxes_list, conf_threshold=0.5)

    cv2.imshow("Face Detection Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()