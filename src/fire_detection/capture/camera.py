import queue
import threading
import time
from collections import deque

import cv2
import numpy as np

_MAX_CONSECUTIVE_FAILURES = 5


class CaptureThread(threading.Thread):
    """
    Continuously reads frames from a webcam into a single-slot latest-frame queue.
    Older unread frames are replaced when inference has not consumed them.
    """

    def __init__(self, device: int, frame_queue: "queue.Queue[np.ndarray]", stop_event: threading.Event):
        super().__init__(daemon=True, name="CaptureThread")
        self._device = device
        self._queue = frame_queue
        self._stop = stop_event

        # Set by run() before blocking on cap.read(); main thread waits on this.
        self.camera_ready = threading.Event()
        self.camera_error: str = ""

        # Written only by this thread; read by main thread for display.
        # No lock: single writer, display-only reads tolerate mild staleness.
        self.frames_captured: int = 0
        self.frames_dropped: int = 0  # frames replaced in queue before inference consumed them
        self._fps_ts: deque = deque(maxlen=30)

    @property
    def capture_fps(self) -> float:
        ts = self._fps_ts
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 0.0

    def run(self) -> None:
        cap = cv2.VideoCapture(self._device)

        if not cap.isOpened():
            self.camera_error = f"Could not open camera (device={self._device})"
            self.camera_ready.set()
            return

        self.camera_ready.set()
        consecutive_failures = 0

        try:
            while not self._stop.is_set():
                ok, frame = cap.read()

                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                        break
                    continue

                consecutive_failures = 0
                self.frames_captured += 1
                self._fps_ts.append(time.monotonic())
                self._put_latest(frame)
        finally:
            cap.release()

    def _put_latest(self, frame: np.ndarray) -> None:
        """Replace queue contents with the newest frame, discarding any unread frame."""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()  # discard stale frame
            except queue.Empty:
                pass  # main thread consumed it between our check and get — queue is now empty
            self.frames_dropped += 1
            self._queue.put_nowait(frame)
