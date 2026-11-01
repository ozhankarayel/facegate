"""Donanım benchmark'ı: model x tespit boyutu x CPU thread sayısı."""
import argparse
import gc
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil
from insightface.app import FaceAnalysis

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def load_sample(path: Path) -> np.ndarray:
    """Örnek kare yoksa webcam'den çeker. (Türkçe karakterli yollar için imread/imwrite kullanılmaz.)"""
    if not path.exists():
        print("Örnek kare yok, webcam'den çekiliyor. Kameraya bak (başka uygulama kamerayı kullanmasın)...")
        cap = cv2.VideoCapture(0)
        for _ in range(15):          # pozlama otursun
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("Kameradan görüntü alınamadı.")
        ok, buf = cv2.imencode(".jpg", frame)
        path.write_bytes(buf.tobytes())
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("Örnek kare okunamadı.")
    return cv2.resize(img, (640, 480))


def hardware_info() -> dict:
    return {
        "cpu": platform.processor(),
        "os": platform.platform(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "python": platform.python_version(),
    }


def build_app(model: str, det: int, threads: int, modules=None):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    t0 = time.perf_counter()
    app = FaceAnalysis(name=model, providers=["CPUExecutionProvider"],
                       sess_options=so, allowed_modules=modules)
    app.prepare(ctx_id=-1, det_size=(det, det))
    return app, time.perf_counter() - t0


def measure(app, img, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        app.get(img)

    proc = psutil.Process()
    cpu0 = sum(proc.cpu_times()[:2])
    wall0 = time.perf_counter()
    lat, found = [], 0
    for _ in range(iters):
        t = time.perf_counter()
        faces = app.get(img)
        lat.append((time.perf_counter() - t) * 1000)
        found += len(faces) > 0
    wall = time.perf_counter() - wall0
    cpu_pct = (sum(proc.cpu_times()[:2]) - cpu0) / wall * 100

    lat = np.array(lat)
    return {
        "mean_ms": round(float(lat.mean()), 1),
        "p50_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "fps": round(1000 / float(lat.mean()), 2),
        "cpu_pct": round(cpu_pct, 0),
        "rss_mb": round(proc.memory_info().rss / 1e6, 0),
        "face_found_rate": round(found / iters, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["buffalo_s", "buffalo_l"])
    ap.add_argument("--det-sizes", nargs="+", type=int, default=[320, 640, 1280])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--image", default=str(HERE / "sample_frame.jpg"))
    ap.add_argument("--quick", action="store_true", help="hızlı deneme: az iterasyon, az ayar")
    ap.add_argument("--modules", nargs="+", default=None,
                    help="örn: --modules detection recognition")
    args = ap.parse_args()

    info = hardware_info()
    (RESULTS / "hw_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Donanım:", json.dumps(info, indent=2, ensure_ascii=False))

    n = info["logical_cores"] or 1
    threads_list = sorted({1, 2, 4, 6, n} & set(range(1, n + 1)))
    if args.quick:
        args.iters, args.det_sizes = 10, [640]
        threads_list = sorted({1, n})

    img = load_sample(Path(args.image))
    _app, _ = build_app("buffalo_s", 640, 4)
    if not _app.get(img):
        raise SystemExit("Örnek karede yüz bulunamadı. sample_frame.jpg dosyasını silip, "
                         "yüzün net ve aydınlık göründüğü bir kareyle yeniden çalıştır.")
    del _app
    rows = []
    out = RESULTS / ("hw_benchmark_lean.csv" if args.modules else "hw_benchmark.csv")

    for model in args.models:
        for det in args.det_sizes:
            for th in threads_list:
                app, load_s = build_app(model, det, th, args.modules)
                r = measure(app, img, args.warmup, args.iters)
                row = {"model": model, "det_size": det, "threads": th,
                       "load_s": round(load_s, 1), **r}
                rows.append(row)
                pd.DataFrame(rows).to_csv(out, index=False)   # her ayardan sonra kaydet
                print(f"{model:10} det={det:<5} thr={th:<2} "
                      f"ort={r['mean_ms']:>7.1f} ms  p95={r['p95_ms']:>7.1f} ms  "
                      f"FPS={r['fps']:>5.2f}  CPU={r['cpu_pct']:>4.0f}%  "
                      f"RAM={r['rss_mb']:>5.0f} MB  yuz={r['face_found_rate']:.0%}")
                del app
                gc.collect()

    print(f"\nKaydedildi: {out}")


if __name__ == "__main__":
    main()