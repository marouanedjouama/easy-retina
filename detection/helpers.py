import cv2
import numpy as np


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bounding boxes with shape (n, 4).
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to keypoints.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded keypoints with shape (n, 2k).
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def draw_detections(image, detections, conf_threshold=0.5):

    dimg = image.copy()

    for det in detections:


        x1, y1, x2, y2, score = det["box"]
        if score < conf_threshold:
            continue

        # Convert to int for drawing
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        

        if det["is_real"] is not None and not det["is_real"]:
            color= (0, 0, 255)
        else:
            color= (0, 255, 0)


        # Draw rectangle
        cv2.rectangle(dimg, (x1, y1), (x2, y2), color, 1)

        label = f"{score:.2f}"
        if det["is_real"] is not None:
            label += f" | {'real' if det['is_real'] else 'fake'}"


        if det["predicted_sex"] is not None:
            label += f" | {det['predicted_sex']}"

        cv2.putText(dimg, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return dimg


def blur_faces(frame, detections, blocks=15, padding=0.2, conf_threshold=0.5):

    for det in detections:
        
        x1, y1, x2, y2, score = det["box"]
        if score < conf_threshold:
            continue

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Compute width/height
        w = x2 - x1
        h = y2 - y1

        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(frame.shape[1], x2 + pad_w)
        y2 = min(frame.shape[0], y2 + pad_h)

        if x2 <= x1 or y2 <= y1:
            continue

        # Extract ROI
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        h_roi, w_roi = roi.shape[:2]

        # Pixelate: resize down and back up
        temp = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)

        # Create oval mask
        mask = np.zeros_like(roi, dtype=np.uint8)
        center = (roi.shape[1] // 2, roi.shape[0] // 2)
        axes = (roi.shape[1] // 2, roi.shape[0] // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)

        # Blend original and pixelated with oval mask
        roi = np.where(mask > 0, pixelated, roi)

        # Replace back
        frame[y1:y2, x1:x2] = roi
        
    return frame