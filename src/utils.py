import numpy as np
from typing import Tuple


def is_point_in_bbox(point: Tuple[float, float], bbox: list) -> bool:
    u, v = point
    x1, y1, x2, y2 = bbox
    return (x1 <= u <= x2) and (y1 <= v <= y2)

def homogeneous(point: np.ndarray) -> np.ndarray:
    return np.append(point, 1.0)


def from_homogeneous(point: np.ndarray) -> np.ndarray:
    return point[:-1] / point[-1]


def compute_bbox_area(bbox: list) -> float:
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm