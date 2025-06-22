#!/usr/bin/env python3
"""Redraw every figure from artifacts/ alone. No GPU and no raw dataset needed.

Reads artifacts/{metrics.json, losses.json, scores_*.npz} and, only for the
signal/FMM examples, data/data_processed/data_processed.npz.
Writes artifacts/figures/ and copies them into the English documents.
"""
import os, sys, json, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, precision_recall_curve

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.data_utils import array_to_fmm_dict

ART = os.path.join(REPO, "artifacts")
FIG = os.path.join(ART, "figures")
os.makedirs(FIG, exist_ok=True)

M = json.load(open(os.path.join(ART, "metrics.json")))
POSTER_BLUE = (66/255, 67/255, 134/255)          # udesa as baposter renders it
poster_cmap = LinearSegmentedColormap.from_list("udesa", [(1, 1, 1), POSTER_BLUE])
CLASSES = ["Normal", "Anomalous"]


def plot_cm(cm, title, cmap, path):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6.9, 5.9))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=cm.max())
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=11)
    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel("Prediction", fontsize=14); ax.set_ylabel("True Label", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES, fontsize=13)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES, fontsize=13, rotation=90, va="center")
    thr = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center", fontsize=20,
                    color="white" if cm[i, j] > thr else "black")
    fig.tight_layout(); fig.savefig(path, dpi=100, facecolor="white"); plt.close(fig)


plot_cm(M["cm_test"], "Confusion Matrix - Test", "Blues", f"{FIG}/CM_dev_test.png")
plot_cm(M["cm_val"], "Confusion Matrix - Validation", "Blues", f"{FIG}/CM_tr_val.png")
plot_cm(M["cm_test"], "Confusion Matrix - Test", poster_cmap, f"{FIG}/matconftest2.png")
plot_cm(M["cm_val"], "Confusion Matrix - Validation", poster_cmap, f"{FIG}/matconfval2.png")

s = np.load(os.path.join(ART, "scores_test.npz"))
yt, ps = s["y_true"], s["y_score"]
fpr, tpr, _ = roc_curve(yt, ps)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="slateblue", lw=4, label=f"AUC = {M['test']['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1.5)
plt.xlabel("FPR", fontsize=14); plt.ylabel("TPR (Recall)", fontsize=14)
plt.title("ROC Curve", fontsize=16); plt.legend(fontsize=14)
plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"{FIG}/ROC_dev_test.png", dpi=100); plt.close()

pr, rc, _ = precision_recall_curve(yt, ps)
plt.figure(figsize=(6, 4))
plt.plot(rc, pr, color="palevioletred", lw=4)
plt.xlabel("Recall", fontsize=14); plt.ylabel("Precision", fontsize=14)
plt.title("Precision-Recall Curve", fontsize=16)
plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"{FIG}/PR_dev_test.png", dpi=100); plt.close()

L = json.load(open(os.path.join(ART, "losses.json")))
ep = range(1, len(L["total"]) + 1)
plt.figure(figsize=(6, 4))
plt.plot(ep, L["total"], color="cornflowerblue", lw=3.5, label="Total Loss (ELBO)")
plt.plot(ep, L["recon"], color="salmon", lw=2.5, label="Recon Loss (MSE)")
plt.plot(ep, L["kl"], color="seagreen", lw=2.5, label="KL Loss")
plt.title("Loss Evolution during Training", fontsize=16, pad=12)
plt.xlabel("Epoch", fontsize=14); plt.ylabel("Loss Value", fontsize=14)
plt.legend(fontsize=13); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{FIG}/TrainLoss.png", dpi=100); plt.close()

# ---------------------------------------------------- signal + FMM examples
NPZ = os.path.join(REPO, "data", "data_processed", "data_processed.npz")
if os.path.exists(NPZ):
    d = np.load(NPZ)
    S, C = d["X_dev_signal_raw"], d["X_dev_coeffs_raw"]
    MAIN = "#4C72B0"
    WC = {"P": "#1f77b4", "Q": "#ff7f0e", "R": "#2ca02c", "S": "#d62728", "T": "#9467bd"}
    SEQ, FS = 2048, 100

    def recon(c):
        p = array_to_fmm_dict(c, num_leads=1)
        t = np.linspace(0, 2*np.pi, SEQ); w = {}
        for k in ["P", "Q", "R", "S", "T"]:
            w[k] = p[k]["A"][0]*np.cos(p[k]["beta"][0] + 2*np.arctan(p[k]["omega"][0]*np.tan((t-p[k]["alpha"][0])/2)))
        return p["P"]["M"][0] + sum(w.values()), w

    # fixed beats (verified clean morphology): healthy 15897, anomalous 1747
    for idx, lab, f1, f2 in [(15897, "Normal", "senal_sana.png", "FMM_sano.png"),
                             (1747, "Anomalous", "senal_anomala.png", "FMM_anomalo.png")]:
        s = S[idx, :, 0]
        L = int(np.max(np.nonzero(np.abs(s) > 1e-9))) + 1  # crop the zero padding
        x = np.linspace(0, 1.1 * L / 128, L)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, s[:L], color=MAIN, lw=3, label=f"{lab} - Sample {idx}")
        ax.set_xlabel("Time (s)", fontsize=16); ax.set_ylabel("Amplitude", fontsize=16)
        ax.set_title(f"{lab} - Sample {idx}", fontsize=18, fontweight="bold")
        ax.title.set_bbox(dict(facecolor="lavenderblush", edgecolor="none", boxstyle="round,pad=0.5", alpha=0.7))
        ax.legend(fontsize=13); ax.grid(alpha=0.2); fig.tight_layout()
        fig.savefig(f"{FIG}/{f1}", dpi=100); plt.close(fig)

        r, w = recon(C[idx])
        t = np.linspace(0, 2 * np.pi, SEQ)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(t, r, color=MAIN, lw=3, label="FMM reconstruction")
        for k, v in w.items():
            ax.plot(t, v, "--", color=WC[k], lw=2, label=f"{k} wave")
        ax.set_xlabel("Time (s)", fontsize=16); ax.set_ylabel("Amplitude", fontsize=16)
        ax.set_title("FMM reconstruction with waves", fontsize=18, fontweight="bold")
        ax.title.set_bbox(dict(facecolor="lavenderblush", edgecolor="none", boxstyle="round,pad=0.5", alpha=0.7))
        ax.legend(fontsize=13); ax.grid(alpha=0.2); fig.tight_layout()
        fig.savefig(f"{FIG}/{f2}", dpi=100); plt.close(fig)

    # NOTE: the poster's señalesconFMM_conlabels.png is curated by hand; not regenerated here.

# ---------------------------------------------------- copy into the documents
REPORT = os.path.join(REPO, "docs", "Informe_EN", "Graficos")
POSTER = os.path.join(REPO, "docs", "poster_EN", "ourfigures")
for f in ["CM_dev_test.png", "CM_tr_val.png", "ROC_dev_test.png", "PR_dev_test.png", "TrainLoss.png"]:
    shutil.copy(f"{FIG}/{f}", os.path.join(REPORT, "Metricas", f))
for f in ["senal_sana.png", "senal_anomala.png", "FMM_sano.png", "FMM_anomalo.png"]:
    shutil.copy(f"{FIG}/{f}", os.path.join(REPORT, "Visualizacion", f))
for f in ["matconftest2.png", "matconfval2.png"]:
    shutil.copy(f"{FIG}/{f}", os.path.join(POSTER, f))
print("figures regenerated and copied into docs/")
