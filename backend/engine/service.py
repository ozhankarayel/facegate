from functools import lru_cache

import cv2
import numpy as np
from django.conf import settings

from .pipeline import FaceEngine


@lru_cache(maxsize=1)
def get_engine() -> FaceEngine:
    """Model ilk çağrıda yüklenir, sonra aynı nesne kullanılır."""
    return FaceEngine(
        model_name=settings.FACE_MODEL_NAME,
        det_size=settings.FACE_DET_SIZE,
    )


def decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Geçersiz görüntü dosyası.")
    return img


def bbox_area(face: dict) -> int:
    x1, y1, x2, y2 = face["bbox"]
    return max(0, x2 - x1) * max(0, y2 - y1)