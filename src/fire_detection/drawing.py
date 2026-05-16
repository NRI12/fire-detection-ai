import cv2
import numpy as np

from fire_detection.inference.schema import Detection


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return annotated
