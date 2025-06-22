#!/usr/bin/env python3
"""Evaluate the trained pipeline on validation and test, and render the figures.

Loads the model, XGB and threshold saved by 02_train.py (VAE on train normals,
XGB on the train split, threshold chosen on validation for recall >= target).
The same threshold is applied to validation and to the held-out test set; the
test numbers are the ones that reflect performance on unseen recordings.

Writes small, committable artifacts (scores, metrics, figures) so every figure
can be redrawn later without the multi-GB dataset.
"""
import os, sys, json
import numpy as np
import torch, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             precision_score, recall_score, f1_score,
                             accuracy_score, confusion_matrix)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.models.VAE import VAE1D, compute_scores
from src.data_utils import array_to_fmm_dict
from src.data_spitting import split_train_val_with_raw_dev

NPZ = os.path.join(REPO, "data", "data_processed", "data_processed.npz")
ART = os.path.join(REPO, "artifacts"); MOD = os.path.join(REPO, "saved_models_and_params")
FIG = os.path.join(ART, "figures"); os.makedirs(FIG, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)

hps = json.load(open(os.path.join(MOD, "best_hps.json")))
beta, bs = hps["beta"], hps["batch_size"]
model = VAE1D(input_ch=1, coeff_ch=21, latent_dim=hps["latent_dim"], n_blocks=hps["n_blocks"])
model.load_state_dict(torch.load(os.path.join(MOD, "vae_state.pth"), map_location=device))
model.to(device).eval()
clf = joblib.load(os.path.join(MOD, "xgb_clf.joblib"))
best_thresh = json.load(open(os.path.join(MOD, "threshold.json")))["best_thresh"]

d = np.load(NPZ)
mean_x, std_x = float(d["mean_x"]), float(d["std_x"])
mean_c, std_c = d["mean_c"], d["std_c"]
ncid = int(d["normal_class_id"])


def score(sig_np, coe_np):
    s = torch.tensor(sig_np).permute(0, 2, 1).float()
    c = torch.tensor(coe_np).float()
    ld = DataLoader(TensorDataset(s, c, torch.zeros(len(s))), batch_size=bs)
    e, z = compute_scores(model, ld, device, beta=beta)
    e = np.nan_to_num(e, nan=0.0, posinf=1e6, neginf=-1e6)
    z = np.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.hstack([e.reshape(-1, 1), z])


def balanced(sig, coe, y):
    m = (y == ncid)
    ns, nc = sig[m], coe[m]
    as_, ac = sig[~m], coe[~m]
    n = min(len(ns), len(as_))
    return np.concatenate([ns[:n], as_[:n]]), np.concatenate([nc[:n], ac[:n]]), \
        np.concatenate([np.zeros(n), np.ones(n)])


def metrics_of(yt, prob, thr):
    yp = (prob >= thr).astype(int)
    return {"roc_auc": float(roc_auc_score(yt, prob)),
            "precision": float(precision_score(yt, yp)),
            "recall": float(recall_score(yt, yp)),
            "f1": float(f1_score(yt, yp)),
            "accuracy": float(accuracy_score(yt, yp))}, confusion_matrix(yt, yp)


results = {}

# ---------------------------------------------------- VALIDATION (same split as training)
ds, dc, yd, rd = d["X_dev_signal_raw"], d["X_dev_coeffs_raw"], d["y_dev"], d["records_dev"]
m = (yd == ncid)
_, _, _, _, _, sig_vn, coe_vn, sig_va, coe_va, _ = split_train_val_with_raw_dev(
    ds[m], dc[m], rd[m], ds[~m], dc[~m], rd[~m], val_frac=0.2, seed=42)
del ds, dc
nv = min(len(sig_vn), len(sig_va))
S = np.concatenate([sig_vn[:nv], sig_va[:nv]]); C = np.concatenate([coe_vn[:nv], coe_va[:nv]])
y_val = np.concatenate([np.zeros(nv), np.ones(nv)])
p_val = clf.predict_proba(score(S, C))[:, 1]
results["validation"], cm_val = metrics_of(y_val, p_val, best_thresh)
np.savez_compressed(os.path.join(ART, "scores_val.npz"), y_true=y_val, y_score=p_val)
print("VALIDATION:", results["validation"], flush=True)

# ---------------------------------------------------- TEST (held out)
Xs = (d["X_test_signal_raw"] - mean_x) / (std_x + 1e-8)
Xc = (d["X_test_coeffs_raw"] - mean_c) / (std_c + 1e-8)
S, C, y_te = balanced(Xs, Xc, d["y_test"])
del Xs, Xc
p_te = clf.predict_proba(score(S, C))[:, 1]
results["test"], cm_te = metrics_of(y_te, p_te, best_thresh)
np.savez_compressed(os.path.join(ART, "scores_test.npz"), y_true=y_te, y_score=p_te)
print("TEST:", results["test"], flush=True)

results["threshold"] = best_thresh
results["criterion"] = json.load(open(os.path.join(MOD, "threshold.json"))).get("criterion", "")
results["cm_test"] = cm_te.tolist(); results["cm_val"] = cm_val.tolist()
json.dump(results, open(os.path.join(ART, "metrics.json"), "w"), indent=1)

# ================================================================== FIGURES
POSTER_BLUE = (66/255, 67/255, 134/255)   # udesa as rendered by baposter
poster_cmap = LinearSegmentedColormap.from_list("udesa", [(1, 1, 1), POSTER_BLUE])
CLASSES = ["Normal", "Anomalous"]


def plot_cm(cm, title, cmap, path):
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
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", fontsize=20,
                    color="white" if cm[i, j] > thr else "black")
    fig.tight_layout(); fig.savefig(path, dpi=100, facecolor="white"); plt.close(fig)


plot_cm(cm_te, "Confusion Matrix - Test", "Blues", f"{FIG}/CM_dev_test.png")
plot_cm(cm_val, "Confusion Matrix - Validation", "Blues", f"{FIG}/CM_tr_val.png")
plot_cm(cm_te, "Confusion Matrix - Test", poster_cmap, f"{FIG}/matconftest2.png")
plot_cm(cm_val, "Confusion Matrix - Validation", poster_cmap, f"{FIG}/matconfval2.png")

fpr, tpr, _ = roc_curve(y_te, p_te)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="slateblue", lw=4, label=f"AUC = {results['test']['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1.5)
plt.xlabel("FPR", fontsize=14); plt.ylabel("TPR (Recall)", fontsize=14)
plt.title("ROC Curve", fontsize=16); plt.legend(fontsize=14)
plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"{FIG}/ROC_dev_test.png", dpi=100); plt.close()

pr, rc, _ = precision_recall_curve(y_te, p_te)
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

# ----------------------------------------------------- signal + FMM examples
MAIN = "#4C72B0"
WC = {"P": "#1f77b4", "Q": "#ff7f0e", "R": "#2ca02c", "S": "#d62728", "T": "#9467bd"}
SEQ, FS = 2048, 100
sig_raw, coe_raw = d["X_dev_signal_raw"], d["X_dev_coeffs_raw"]
IDX_NORM, IDX_ANO = 15897, 1747   # verified clean beats


def fmm_recon(c):
    p = array_to_fmm_dict(c, num_leads=1)
    t = np.linspace(0, 2*np.pi, SEQ); w = {}
    for k in ["P", "Q", "R", "S", "T"]:
        w[k] = p[k]["A"][0]*np.cos(p[k]["beta"][0] + 2*np.arctan(p[k]["omega"][0]*np.tan((t-p[k]["alpha"][0])/2)))
    return p["P"]["M"][0] + sum(w.values()), w


for idx, lab, f1n, f2n in [(IDX_NORM, "Normal", "senal_sana.png", "FMM_sano.png"),
                           (IDX_ANO, "Anomalous", "senal_anomala.png", "FMM_anomalo.png")]:
    s = sig_raw[idx, :, 0]
    Lp = int(np.max(np.nonzero(np.abs(s) > 1e-9))) + 1
    x = np.linspace(0, 1.1 * Lp / 128, Lp)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, s[:Lp], color=MAIN, lw=3, label=f"{lab} - Sample {idx}")
    ax.set_xlabel("Time (s)", fontsize=16); ax.set_ylabel("Amplitude", fontsize=16)
    ax.set_title(f"{lab} - Sample {idx}", fontsize=18, fontweight="bold")
    ax.title.set_bbox(dict(facecolor="lavenderblush", edgecolor="none", boxstyle="round,pad=0.5", alpha=0.7))
    ax.legend(fontsize=13); ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(f"{FIG}/{f1n}", dpi=100); plt.close(fig)

    rec, waves = fmm_recon(coe_raw[idx])
    t = np.linspace(0, 2*np.pi, SEQ)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, rec, color=MAIN, lw=3, label="FMM reconstruction")
    for k, wv in waves.items():
        ax.plot(t, wv, "--", color=WC[k], lw=2, label=f"{k} wave")
    ax.set_xlabel("Time (s)", fontsize=16); ax.set_ylabel("Amplitude", fontsize=16)
    ax.set_title("FMM reconstruction with waves", fontsize=18, fontweight="bold")
    ax.title.set_bbox(dict(facecolor="lavenderblush", edgecolor="none", boxstyle="round,pad=0.5", alpha=0.7))
    ax.legend(fontsize=13); ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(f"{FIG}/{f2n}", dpi=100); plt.close(fig)

print("DONE_EVAL", flush=True)
