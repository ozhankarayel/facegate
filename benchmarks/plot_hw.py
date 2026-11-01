"""hw_benchmark CSV'sinden grafik ve tablo üretir. Kullanım: python plot_hw.py [csv_adı]"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIG = HERE.parent / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
name = sys.argv[1] if len(sys.argv) > 1 else "hw_benchmark"

AUC = {"buffalo_s": 0.9550, "buffalo_l": 0.9692}   # LFW test, det_size=640

df = pd.read_csv(HERE / "results" / f"{name}.csv")
df = df[df["face_found_rate"] == 1.0]              # yüz bulunamayan ayarlar geçersiz
best = df.loc[df.groupby(["model", "det_size"])["mean_ms"].idxmin()]

# 1) Tespit boyutu (her ayar için en iyi thread)
fig, ax = plt.subplots(figsize=(6, 4))
for model, g in best.groupby("model"):
    g = g.sort_values("det_size")
    ax.plot(g["det_size"], g["mean_ms"], "o-", label=f"{model} ort.")
    ax.plot(g["det_size"], g["p95_ms"], "s--", alpha=0.6, label=f"{model} p95")
ax.set_xlabel("Tespit giriş boyutu (px)")
ax.set_ylabel("Gecikme (ms)")
ax.set_title("Tespit boyutu ve gecikme (en iyi thread)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIG / f"{name}_detsize.png", dpi=150)

# 2) Thread ölçeklenmesi (det=640)
fig, ax = plt.subplots(figsize=(6, 4))
for model, g in df[df["det_size"] == 640].groupby("model"):
    g = g.sort_values("threads")
    ax.plot(g["threads"], g["mean_ms"], "o-", label=model)
ax.set_xlabel("CPU thread sayısı")
ax.set_ylabel("Ortalama gecikme (ms)")
ax.set_title("Thread ölçeklenmesi (det=640)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIG / f"{name}_threads.png", dpi=150)

# 3) Hız - doğruluk takası (det=640, en iyi thread)
fig, ax = plt.subplots(figsize=(5.5, 4))
for _, r in best[best["det_size"] == 640].iterrows():
    ax.scatter(r["mean_ms"], AUC[r["model"]], s=80)
    ax.annotate(f"{r['model']} ({int(r['threads'])} thr)", (r["mean_ms"], AUC[r["model"]]),
                textcoords="offset points", xytext=(6, 6))
ax.set_xlabel("Ortalama gecikme (ms), düşük daha iyi")
ax.set_ylabel("LFW AUC, yüksek daha iyi")
ax.set_title("Hız - doğruluk takası")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / f"{name}_tradeoff.png", dpi=150)

print(df.sort_values(["model", "det_size", "threads"]).to_string(index=False))
print(f"\nGrafikler: {FIG}")