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
    """
    Draw bounding boxes from face detection results.
    
    Args:
        image (np.ndarray): Input image (BGR).
        detections (list or np.ndarray): List/array of detections [x1, y1, x2, y2, score].
        conf_threshold (float): Minimum confidence score to draw a box.
        
    Returns:
        np.ndarray: Image with drawn bounding boxes.
    """
    dimg = image.copy()


    for det in detections:
        is_real, spoof_score = True, 0.0
        if isinstance(det, list):
            x1, y1, x2, y2, score = det
        
        else:
            box, is_real, spoof_score = det
            x1, y1, x2, y2, score = box
        if score < conf_threshold:
            continue

        # Convert to int for drawing
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        

        if is_real:
            color= (0, 255, 0)
        else:
            color= (0, 0, 255)

        # Draw rectangle
        cv2.rectangle(dimg, (x1, y1), (x2, y2), color, 1)
        label = f"{score:.2f} | {'real' if is_real else 'fake'} | score: {spoof_score:.2f}"
        cv2.putText(dimg, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return dimg