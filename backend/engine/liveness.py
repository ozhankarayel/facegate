from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"

# Face Mesh göz noktaları (sıra: p1..p6). Yeni modelde de aynı indeksler geçerli.
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


def _ear(pts: np.ndarray) -> float:
    p1, p2, p3, p4, p5, p6 = pts
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return float(vertical / (2.0 * horizontal)) if horizontal > 0 else 0.0


class EarExtractor:
    """Her kareyi bağımsız işler. İstek başına bir örnek oluştur."""

    def __init__(self):
        # Yol yerine dosyayı bayt olarak veriyoruz: "Masaüstü" gibi Türkçe
        # karakterli yollarda MediaPipe dosyayı açamayabiliyor.
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def ear(self, bgr_image):
        """İki gözün ortalama EAR değeri. Yüz bulunamazsa None."""
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect(mp_image)
        if not res.face_landmarks:
            return None
        h, w = bgr_image.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
        return (_ear(pts[RIGHT_EYE]) + _ear(pts[LEFT_EYE])) / 2.0

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def detect_blink(ears, drop_ratio=0.70, min_open=0.18, min_valid=6):
    """
    EAR dizisinde 'açık -> kapalı -> açık' örüntüsü arar.
    Taban (açık göz) değeri, EAR'ların 80. yüzdeliğidir.
    """
    valid = [e for e in ears if e is not None]
    info = {"blink": False, "ear_baseline": None, "ear_min": None, "frames_used": len(valid)}
    if len(valid) < min_valid:
        return info

    baseline = float(np.percentile(valid, 80))
    info["ear_baseline"] = round(baseline, 3)
    info["ear_min"] = round(float(min(valid)), 3)
    if baseline < min_open:      # gözler zaten kısık/kapalı, güvenilir ölçüm yok
        return info

    closed = [e < baseline * drop_ratio for e in valid]
    for i, is_closed in enumerate(closed):
        if is_closed and any(not c for c in closed[:i]) and any(not c for c in closed[i + 1:]):
            info["blink"] = True
            break
    return info