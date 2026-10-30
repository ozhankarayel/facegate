import numpy as np
from insightface.app import FaceAnalysis


class FaceEngine:
    def __init__(self, model_name="buffalo_s", det_size=(640, 640), providers=None):
        providers = providers or ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=-1, det_size=det_size)

    def extract(self, bgr_image):
        """Görüntüdeki her yüz için bbox, tespit skoru ve normalize embedding döndürür."""
        return [
            {
                "bbox": f.bbox.astype(int).tolist(),
                "score": float(f.det_score),
                "embedding": f.normed_embedding,
            }
            for f in self.app.get(bgr_image)
        ]


def cosine_similarity(a, b):
    # embedding'ler zaten normalize olduğu için nokta çarpımı yeterli
    return float(np.dot(a, b))


def best_match(embedding, gallery, threshold=0.4):
    """gallery: {isim: embedding}. (isim, skor) döner, eşleşme yoksa (None, skor)."""
    best_name, best_score = None, -1.0
    for name, ref in gallery.items():
        s = cosine_similarity(embedding, ref)
        if s > best_score:
            best_name, best_score = name, s
    return (best_name if best_score >= threshold else None), best_score