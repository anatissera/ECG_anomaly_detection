#!/usr/bin/env python3
"""Train the pipeline with an honest train/validation/test protocol.

  * split the dev set into train and validation, grouped by recording
  * VAE  -> trained on the TRAIN normals only
  * XGB  -> trained on the TRAIN beats (normals + train anomalies)
  * threshold -> chosen on VALIDATION (unseen by the VAE and the XGB) as the
    highest cut that keeps recall >= TARGET_RECALL on the anomalous class

Nothing here ever touches the test set. Saves to saved_models_and_params/ the
model, the XGB, the threshold and the hyperparameters, plus artifacts/losses.json.
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn, joblib
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.models.VAE import VAE1D, compute_scores
from src.data_spitting import split_train_val_with_raw_dev
from src.thresholds import threshold_for_recall

NPZ = os.path.join(REPO, "data", "data_processed", "data_processed.npz")
ART = os.path.join(REPO, "artifacts")
MOD = os.path.join(REPO, "saved_models_and_params")
os.makedirs(MOD, exist_ok=True); os.makedirs(ART, exist_ok=True)

HPS = {"beta": 1.0, "lr": 5e-4, "latent_dim": 64, "n_blocks": 3,
       "batch_size": 64, "epochs": 35}
TARGET_RECALL = 0.80
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)
torch.manual_seed(0); np.random.seed(0)

d = np.load(NPZ)
ds, dc, yd, rd = d["X_dev_signal_raw"], d["X_dev_coeffs_raw"], d["y_dev"], d["records_dev"]
ncid = int(d["normal_class_id"])
m = (yd == ncid)

# -------------------------------------------------- grouped train/val split
train_ds, val_ds, stats, sig_tr, coe_tr, sig_vn, coe_vn, sig_va, coe_va, idx_val_ano = \
    split_train_val_with_raw_dev(ds[m], dc[m], rd[m], ds[~m], dc[~m], rd[~m],
                                 val_frac=0.2, seed=42)
bs, beta = HPS["batch_size"], HPS["beta"]
print(f"train normals {len(sig_tr)}, val normals {len(sig_vn)}, val anomalies {len(sig_va)}",
      flush=True)

# -------------------------------------------------- VAE on train normals
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
model = VAE1D(input_ch=1, coeff_ch=train_ds.tensors[1].shape[1],
              latent_dim=HPS["latent_dim"], n_blocks=HPS["n_blocks"]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=HPS["lr"])
recon_hist, kl_hist, total_hist = [], [], []
print("Starting VAE training...", flush=True)
for ep in range(HPS["epochs"]):
    model.train(); t0 = time.time(); er = ek = 0.0
    for xs, xc, _ in train_loader:
        xs, xc = xs.to(device), xc.to(device)
        rec, mu, logv = model(xs, xc)
        recon_loss = nn.functional.mse_loss(rec, xs)
        kl_loss = (-0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum()) / xs.size(0)
        loss = recon_loss + beta * kl_loss
        opt.zero_grad(); loss.backward(); opt.step()
        er += recon_loss.item() * xs.size(0); ek += kl_loss.item() * xs.size(0)
    N = len(train_loader.dataset)
    recon_hist.append(er / N); kl_hist.append(ek / N)
    total_hist.append(recon_hist[-1] + beta * kl_hist[-1])
    print(f"Epoch {ep+1}/{HPS['epochs']} Recon={recon_hist[-1]:.4f} KL={kl_hist[-1]:.4f} "
          f"Total={total_hist[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
model.eval()
torch.save(model.state_dict(), os.path.join(MOD, "vae_state.pth"))
json.dump({"recon": recon_hist, "kl": kl_hist, "total": total_hist},
          open(os.path.join(ART, "losses.json"), "w"), indent=1)


def feats(sig, coe):
    s = torch.tensor(sig).permute(0, 2, 1).float()
    c = torch.tensor(coe).float()
    e, z = compute_scores(model, DataLoader(TensorDataset(s, c, torch.zeros(len(s))),
                                            batch_size=bs), device, beta)
    e = np.nan_to_num(e, nan=0.0, posinf=1e6, neginf=-1e6)
    z = np.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.hstack([e.reshape(-1, 1), z])

# -------------------------------------------------- XGB on train beats
idx_tr_ano = np.setdiff1d(np.arange((~m).sum()), idx_val_ano)
sig_tr_ano = (ds[~m][idx_tr_ano] - stats['mean_sig']) / stats['std_sig']
coe_tr_ano = (dc[~m][idx_tr_ano] - stats['mean_coe']) / stats['std_coe']
X_train = np.vstack([feats(sig_tr, coe_tr), feats(sig_tr_ano, coe_tr_ano)])
y_train = np.concatenate([np.zeros(len(sig_tr)), np.ones(len(sig_tr_ano))])
print("Training XGBoost on the train split...", flush=True)
clf = XGBClassifier(eval_metric="logloss")
clf.fit(X_train, y_train)
joblib.dump(clf, os.path.join(MOD, "xgb_clf.joblib"))

# -------------------------------------------------- threshold on VALIDATION
nv = min(len(sig_vn), len(sig_va))
p_val = clf.predict_proba(np.vstack([feats(sig_vn[:nv], coe_vn[:nv]),
                                     feats(sig_va[:nv], coe_va[:nv])]))[:, 1]
y_val = np.concatenate([np.zeros(nv), np.ones(nv)])
best_thresh = threshold_for_recall(y_val, p_val, TARGET_RECALL)
from sklearn.metrics import recall_score, roc_auc_score
print(f"val AUC={roc_auc_score(y_val, p_val):.4f}", flush=True)
print(f"threshold for recall>={TARGET_RECALL} on val: {best_thresh:.3f} "
      f"(val recall {recall_score(y_val, (p_val>=best_thresh).astype(int)):.3f})", flush=True)

json.dump({"best_thresh": best_thresh, "criterion": f"recall>={TARGET_RECALL} on validation"},
          open(os.path.join(MOD, "threshold.json"), "w"))
json.dump({**HPS, "target_recall": TARGET_RECALL},
          open(os.path.join(MOD, "best_hps.json"), "w"), indent=1)
print("DONE_TRAIN", flush=True)
