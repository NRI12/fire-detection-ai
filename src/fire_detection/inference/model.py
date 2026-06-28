import time

import numpy as np
import torch
from ultralytics import YOLO

from fire_detection.inference.schema import Detection


class YOLOv8Detector:
    def __init__(self, weights: str = "yolov8n.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._model = YOLO(weights)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{weights}': {e}") from e
        # FP16 chỉ hợp lệ trên CUDA; CPU sẽ lỗi nếu half=True
        self._half = self.device == "cuda"

    def predict(self, frame: np.ndarray, conf: float = 0.45, imgsz: int = 640) -> tuple[list[Detection], float]:
        t0 = time.monotonic()
        results = self._model.predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            device=self.device,
            half=self._half,
            verbose=False,
        )
        elapsed = time.monotonic() - t0

        detections = self._extract(results)
        del results  # Results objects giữ reference tới frame, xóa ngay để tránh rò bộ nhớ
        return detections, elapsed

    def _extract(self, results) -> list[Detection]:
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append(Detection(
                    box=(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0]),
                    class_name=names[int(box.cls[0])],
                ))
        return detections
