"""LFW çiftleri için embedding çıkarır ve benchmarks/results/ altına önbellekler."""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_pairs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "engine"))
from pipeline import FaceEngine  # noqa: E402

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def to_bgr(img: np.ndarray) -> np.ndarray:
    """sklearn RGB float döndürür (genelde 0-1). OpenCV için uint8 BGR'ye çevir."""
    if img.max() <= 1.0:
        img = img * 255.0
    rgb = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def best_face_embedding(engine, bgr):
    faces = engine.extract(bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: f["score"])
    return face["embedding"]


def run(model: str, subset: str, limit: int | None, det_size: int):
    print(f"[{subset}] LFW çiftleri indiriliyor/yükleniyor...")
    data = fetch_lfw_pairs(
        subset=subset,
        color=True,
        resize=1.0,
        slice_=(slice(0, 250), slice(0, 250)),  # tam 250x250 görüntü
    )
    pairs, labels = data.pairs, data.target
    if limit:
        pairs, labels = pairs[:limit], labels[:limit]

    engine = FaceEngine(model_name=model, det_size=(det_size, det_size))
    n = len(pairs)
    emb = np.zeros((n, 2, 512), dtype=np.float32)
    valid = np.zeros((n, 2), dtype=bool)

    t0 = time.time()
    for i in range(n):
        for j in range(2):
            e = best_face_embedding(engine, to_bgr(pairs[i, j]))
            if e is not None:
                emb[i, j] = e
                valid[i, j] = True
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}  ({time.time() - t0:.0f} sn)")

    out = RESULTS / f"lfw_{model}_{subset}.npz"
    np.savez_compressed(out, emb=emb, valid=valid, labels=labels)
    fta = 1.0 - valid.mean()
    print(f"Kaydedildi: {out}  | yüz bulunamayan görüntü oranı (FTA): {fta:.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="buffalo_s")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--limit", type=int, default=None, help="hızlı deneme için çift sayısı")
    args = ap.parse_args()
    for subset in ("train", "test"):
        run(args.model, subset, args.limit, args.det_size)