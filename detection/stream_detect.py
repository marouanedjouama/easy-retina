import os
import cv2
from pathlib import Path
from detect import SCRFD
from anti_spoof.FasNat import Fasnet
from helpers import draw_detections, blur_faces
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Face recognition inference")
    parser.add_argument("--model",type=str, default="models/scrfd_2.5g.onnx", help="Path to the detection model image")
    parser.add_argument("--n-frames",type=int, default=5, help="run the detection model every n frames")
    parser.add_argument(
        "--face-blur", 
        action="store_true", 
        help="Enable face blurring mode (default: disabled)"
    )
    parser.add_argument(
        "--anti-spoof", 
        action="store_true", 
        help="Enable anti spoof mode (default: disabled)"
    )
    return parser.parse_args()


def main(args):

    model_path = args.model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"❌ File not found: {model_path}")

    detector = SCRFD(model_path=model_path)
    if args.anti_spoof:
        antispoof_model = Fasnet()

    cap = cv2.VideoCapture(0)

    frame_count = 0
    boxes_list = []

    # detect every N frames
    N = args.n_frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1


        results = []
        if frame_count % N == 0:  
            boxes_list, _ = detector.detect(frame)

            if args.anti_spoof:
                for box in boxes_list:
                    x, y, w, h, _ = box
                    is_real, spoof_score = antispoof_model.analyze(img=frame, facial_area=(x, y, w, h))
                    results.append((box, is_real, spoof_score))

            else:
                results = boxes_list


        if args.face_blur:
            frame = blur_faces(frame,results)

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