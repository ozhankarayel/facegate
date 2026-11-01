"""ROC, EER ve eşik kalibrasyonu. Eşik train'de seçilir, test'te ölçülür."""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

HERE = Path(__file__).parent
RESULTS = HERE / "results"
FIGURES = HERE.parent / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load(model: str, subset: str):
    d = np.load(RESULTS / f"lfw_{model}_{subset}.npz")
    emb, valid, y = d["emb"], d["valid"], d["labels"]
    scores = np.einsum("nd,nd->n", emb[:, 0], emb[:, 1])   # embedding'ler normalize: nokta çarpımı = kosinüs
    ok = valid.all(axis=1)
    scores = np.where(ok, scores, -1.0)                    # yüz bulunamazsa çift reddedilir
    return scores, y, ok


def eer_threshold(y, scores):
    fpr, tpr, thr = roc_curve(y, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(thr[i]), float((fpr[i] + fnr[i]) / 2)


def far_threshold(y, scores, target_far):
    """FAR <= target_far koşulunu sağlayan en düşük eşik."""
    neg = np.sort(scores[y == 0])
    k = int(np.ceil(len(neg) * (1 - target_far))) - 1
    return float(neg[min(max(k, 0), len(neg) - 1)]) + 1e-6


def evaluate(y, scores, t):
    far = float(np.mean(scores[y == 0] >= t))
    frr = float(np.mean(scores[y == 1] < t))
    acc = float(np.mean((scores >= t) == (y == 1)))
    return far, frr, acc


def main(model: str):
    s_tr, y_tr, ok_tr = load(model, "train")
    s_te, y_te, ok_te = load(model, "test")

    t_eer, eer_train = eer_threshold(y_tr, s_tr)
    t_far1 = far_threshold(y_tr, s_tr, 0.01)

    fpr, tpr, _ = roc_curve(y_te, s_te)
    auc_te = auc(fpr, tpr)
    fta = 1 - np.concatenate([ok_tr, ok_te]).mean()

    print(f"\n=== Model: {model} ===")
    print(f"FTA (çift bazında yüz bulunamama): {fta:.2%}")
    print(f"AUC (test): {auc_te:.4f}")
    print(f"EER (train): {eer_train:.2%}\n")
    print(f"{'Eşik seçimi':<22}{'Eşik':>8}{'FAR':>9}{'FRR':>9}{'Doğruluk':>11}   (test seti)")
    rows = [("Mevcut (sabit 0.40)", 0.40), ("EER noktası", t_eer), ("FAR<=%1", t_far1)]
    for name, t in rows:
        far, frr, acc = evaluate(y_te, s_te, t)
        print(f"{name:<22}{t:>8.3f}{far:>9.2%}{frr:>9.2%}{acc:>11.2%}")

    print("\nEşik taraması (test seti):")
    for t in np.arange(0.20, 0.55, 0.05):
        far, frr, acc = evaluate(y_te, s_te, t)
        print(f"  {t:.2f}  FAR={far:.2%}  FRR={frr:.2%}  Doğruluk={acc:.2%}")

    print("\nEşik taraması (train seti):")
    for t in np.arange(0.20, 0.55, 0.05):
        far, frr, acc = evaluate(y_tr, s_tr, t)
        print(f"  {t:.2f}  FAR={far:.2%}  FRR={frr:.2%}  Doğruluk={acc:.2%}")

    # Grafikler
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(fpr, tpr, lw=2, label=f"AUC={auc_te:.4f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set_xscale("symlog", linthresh=1e-3)
    ax[0].set_xlabel("FAR (yanlış kabul)")
    ax[0].set_ylabel("TAR (doğru kabul)")
    ax[0].set_title(f"ROC (LFW test) - {model}")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    bins = np.linspace(-0.2, 1.0, 60)
    ax[1].hist(s_te[y_te == 1], bins, alpha=0.6, label="Aynı kişi")
    ax[1].hist(s_te[y_te == 0], bins, alpha=0.6, label="Farklı kişi")
    ax[1].axvline(t_eer, color="k", ls="--", label=f"EER eşiği {t_eer:.2f}")
    ax[1].set_xlabel("Kosinüs benzerliği")
    ax[1].set_ylabel("Çift sayısı")
    ax[1].set_title("Skor dağılımı")
    ax[1].legend()
    fig.tight_layout()
    out = FIGURES / f"roc_{model}.png"
    fig.savefig(out, dpi=150)
    print(f"\nGrafik: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="buffalo_s")
    main(ap.parse_args().model)