from dataclasses import dataclass


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
