"""Matlab cilt segmentasyonu (Python portu) ile InsightFace yüz tespitini LFW üzerinde karşılaştırır.
Referans: InsightFace yüz kutusu (gerçek etiket değil, uyum ölçülür)."""
import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.color import rgb2hsv, rgb2ycbcr
from skimage.measure import regionprops
from sklearn.datasets import fetch_lfw_pairs

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "engine"))
from pipeline import FaceEngine  # noqa: E402

RESULTS = Path(__file__).parent / "results"
FIG = ROOT / "docs" / "figures"
RESULTS.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def skin_regions(rgb, cfg):
    """Matlab highPrecisionSkinDetection portu. (kutular[x1,y1,x2,y2], nihai_maske) döner."""
    ycc, hsv = rgb2ycbcr(rgb), rgb2hsv(rgb)
    cb, cr, h, s = ycc[..., 1], ycc[..., 2], hsv[..., 0], hsv[..., 1]
    mask = ((cb >= 80) & (cb <= 125) & (cr >= 135) & (cr <= 180)
            & (h >= 0.0) & (h <= 0.1) & (s >= 0.1) & (s <= 0.7))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg["close"] + 1,) * 2)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)  # imclose
    mask = ndimage.binary_fill_holes(mask)                                           # imfill

    eight = np.ones((3, 3))
    lab, _ = ndimage.label(mask, structure=eight)                                    # bwareaopen
    big = np.bincount(lab.ravel()) >= cfg["min_area"]
    big[0] = False
    mask = big[lab]

    lab, n = ndimage.label(mask, structure=eight)
    final = np.zeros_like(mask)
    if n == 0:
        return [], final
    stats = [r for r in regionprops(lab)
             if r.eccentricity < 0.90 and r.extent < 0.80 and r.area > cfg["min_area"]]
    if not stats:
        return [], final
    stats.sort(key=lambda r: r.area, reverse=True)
    max_area = stats[0].area
    boxes = []
    for r in stats[: cfg["keep_n"]]:
        if r.area > max_area * 0.20:
            final[lab == r.label] = True
            r0, c0, r1, c1 = r.bbox
            boxes.append([c0, r0, c1, r1])
    return boxes, final


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def to_rgb_uint8(img):
    if img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="buffalo_s")
    ap.add_argument("--limit", type=int, default=None, help="hızlı deneme için görüntü sayısı")
    ap.add_argument("--min-area", type=int, default=5000)
    args = ap.parse_args()
    cfg = {"close": 15, "min_area": args.min_area, "keep_n": 3}

    data = fetch_lfw_pairs(subset="test", color=True, resize=1.0,
                           slice_=(slice(0, 250), slice(0, 250)))
    imgs = data.pairs.reshape(-1, *data.pairs.shape[2:])
    if args.limit:
        imgs = imgs[: args.limit]

    engine = FaceEngine(model_name=args.model, det_size=(640, 640), threads=4)
    engine.extract(cv2.cvtColor(to_rgb_uint8(imgs[0]), cv2.COLOR_RGB2BGR))   # ısınma

    rng = np.random.default_rng(0)
    show = set(rng.choice(len(imgs), size=min(8, len(imgs)), replace=False).tolist())
    rows, examples, skipped = [], [], 0

    for i, im in enumerate(imgs):
        rgb = to_rgb_uint8(im)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        t = time.perf_counter()
        faces = engine.extract(bgr)
        t_face = (time.perf_counter() - t) * 1000
        if not faces:
            skipped += 1
            continue
        fb = max(faces, key=lambda f: f["score"])["bbox"]

        t = time.perf_counter()
        boxes, mask = skin_regions(rgb, cfg)
        t_skin = (time.perf_counter() - t) * 1000

        ious = [iou(fb, b) for b in boxes]
        x1, y1, x2, y2 = [max(0, v) for v in fb]
        cov = float(mask[y1:y2, x1:x2].mean()) if (y2 > y1 and x2 > x1) else 0.0
        rows.append({"idx": i, "n_regions": len(boxes), "best_iou": max(ious, default=0.0),
                     "coverage": cov, "ms_face": t_face, "ms_skin": t_skin})

        if i in show:
            vis = rgb.copy()
            cv2.rectangle(vis, (fb[0], fb[1]), (fb[2], fb[3]), (0, 255, 0), 2)
            for b in boxes:
                cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
            examples.append(vis)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"baseline_compare_{args.model}.csv", index=False)

    n = len(df)
    print(f"\n=== Baseline karşılaştırması (LFW test, {n} görüntü, InsightFace {args.model} referans) ===")
    print(f"InsightFace'in yüz bulamadığı, atlanan görüntü: {skipped}")
    print(f"Cilt yönteminin hiç bölge bulamadığı görüntü:   {(df.n_regions == 0).mean():.1%}")
    print(f"Ortalama bölge sayısı (ideal: 1):               {df.n_regions.mean():.2f}")
    print(f"Ortalama IoU (bölge yoksa 0):                   {df.best_iou.mean():.3f}")
    print(f"IoU >= 0.5 (yüz bulundu sayılır):               {(df.best_iou >= 0.5).mean():.1%}")
    print(f"IoU >= 0.3:                                     {(df.best_iou >= 0.3).mean():.1%}")
    print(f"Yüz kutusunun ort. cilt maskesiyle kaplanması:  {df.coverage.mean():.1%}")
    print(f"Ort. süre: InsightFace {df.ms_face.mean():.1f} ms | cilt portu {df.ms_skin.mean():.1f} ms (karşılaştırılamaz)")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(df.best_iou, bins=20, range=(0, 1))
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("En iyi IoU (cilt bölgesi - InsightFace kutusu)")
    ax.set_ylabel("Görüntü sayısı")
    ax.set_title("Baseline uyumu (LFW test)")
    fig.tight_layout()
    fig.savefig(FIG / "baseline_iou.png", dpi=150)

    if examples:
        fig, axs = plt.subplots(2, 4, figsize=(11, 6))
        for a, e in zip(axs.ravel(), examples):
            a.imshow(e)
            a.axis("off")
        fig.suptitle("Yeşil: InsightFace | Mavi: cilt segmentasyonu bölgeleri")
        fig.tight_layout()
        fig.savefig(FIG / "baseline_examples.png", dpi=150)
    print(f"\nGrafikler: {FIG}")


if __name__ == "__main__":
    main()