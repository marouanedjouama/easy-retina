import os
import cv2
from pathlib import Path
from detect import SCRFD
from anti_spoof.FasNat import Fasnet
from helpers import draw_detections, blur_faces
import argparse
from PIL import Image
import numpy as np
import onnxruntime as ort
from torchvision import transforms


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


gender_classes = ["male", "female"]


def setup_gender_model():
    ort_session = ort.InferenceSession("models/mobilenet_gender.onnx")

    # Preprocessing pipeline (same as training)
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return ort_session, transform


def parse_args():
    parser = argparse.ArgumentParser(description="Face recognition inference")
    parser.add_argument("--model",type=str, default="models/scrfd_2.5g.onnx", help="Path to the detection model image")
    parser.add_argument("--n-frames",type=int, default=5, help="run the detection model every n frames")
    parser.add_argument(
        "--face-blur", 
        action="store_true", 
        help="Enable face blurring mode"
    )
    parser.add_argument(
        "--anti-spoof", 
        action="store_true", 
        help="Enable anti spoof mode"
    )
    parser.add_argument(
        "--gender", 
        action="store_true", 
        help="detect gender"
    )
    return parser.parse_args()


def main(args):

    model_path = args.model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"❌ File not found: {model_path}")

    detector = SCRFD(model_path=model_path)
    if args.anti_spoof:
        antispoof_model = Fasnet()


    # gender 
    ort_session, transform = setup_gender_model()

    cap = cv2.VideoCapture(0)

    frame_count = 0
    results = []

    # detect every N frames
    N = args.n_frames
    pad = 30
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % N == 0:  
            results = []
            boxes_list, _ = detector.detect(frame)

            # gender prediction for each box
            fh, fw = frame.shape[:2]  # frame height and width


            for box in boxes_list:

                x1, y1, x2, y2, _ = [int(v) for v in box]

                predicted_sex = None
                if args.gender:
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(fw, x2 + pad)
                    y2 = min(fh, y2 + pad)
                    face = frame[y1:y2, x1:x2]           

                    face = Image.fromarray(face) 
                    face = transform(face).unsqueeze(0)
                    face = face.numpy().astype(np.float32)

                    # Run inference
                    outputs = ort_session.run(None, {"input": face})
                    pred_logits = outputs[0]  # raw predictions
                    pred_class = np.argmax(pred_logits, axis=1)[0]

                    # Map to class label
                    predicted_sex = gender_classes[pred_class]

                is_real = None
                if args.anti_spoof:
                    # face box width and height
                    w = x2 - x1 
                    h = y2 - y1
                    is_real, _ = antispoof_model.analyze(img=frame, facial_area=(x1, y1, w, h))

                results.append({
                    "box": box,
                    "predicted_sex" : predicted_sex,
                    "is_real" : is_real
                })


        if args.face_blur:
            frame = blur_faces(frame,results, blocks=10)

        if len(results):
            frame = draw_detections(frame, results, conf_threshold=0.5)

        cv2.imshow("Face Detection Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)