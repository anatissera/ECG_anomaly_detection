#!/usr/bin/env python3
"""Measure how much the sweep score moves when only the random seed changes.

The hyperparameter sweep picks the configuration with the best validation AUC.
That is only meaningful if the gap between configurations is larger than the
run-to-run noise of training the *same* configuration twice. This script trains
one configuration several times, changing nothing but the seed, and reports the
spread. If the spread is comparable to the gap between the top configurations,
the ranking is mostly noise and enlarging the grid would only buy a luckier
draw, not a better model.

The training/scoring protocol mirrors the sweep in notebooks/pipeline_VAE.ipynb
(15 epochs, XGBoost on [-ELBO, mu]) so the numbers are directly comparable.
"""
import os, sys, json, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.models.VAE import VAE1D, compute_scores
from src.data_spitting import split_train_val_with_raw_dev

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=15)   # same as the sweep
args = ap.parse_args()

NPZ = os.path.join(REPO, "data", "data_processed", "data_processed.npz")
MOD = os.path.join(REPO, "saved_models_and_params")
ART = os.path.join(REPO, "artifacts")

hps = json.load(open(os.path.join(MOD, "best_hps.json")))
beta, lr = hps["beta"], hps["lr"]
ld, nb, bs = int(hps["latent_dim"]), int(hps["n_blocks"]), int(hps["batch_size"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"config under test: beta={beta} lr={lr} latent_dim={ld} n_blocks={nb}", flush=True)
print(f"device: {device}", flush=True)

d = np.load(NPZ)
ds, dc, yd, rd = d["X_dev_signal_raw"], d["X_dev_coeffs_raw"], d["y_dev"], d["records_dev"]
ncid = int(d["normal_class_id"])
m = (yd == ncid)
ano_s, ano_c, ano_r = ds[~m], dc[~m], rd[~m]

# identical grouped split to the notebook (seed 42) so we vary only the training seed
train_ds, val_ds, stats, sig_tr, coe_tr, sig_vn, coe_vn, sig_va, coe_va, idx_val_ano = \
    split_train_val_with_raw_dev(ds[m], dc[m], rd[m], ano_s, ano_c, ano_r,
                                 val_frac=0.2, seed=42)
del ds, dc

idx_tr_ano = np.setdiff1d(np.arange(len(ano_s)), idx_val_ano)
sig_tr_ano = (ano_s[idx_tr_ano] - stats['mean_sig']) / stats['std_sig']
coe_tr_ano = (ano_c[idx_tr_ano] - stats['mean_coe']) / stats['std_coe']
del ano_s, ano_c

train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
labels_val = val_ds.tensors[2].numpy()

tr_norm_s = train_ds.tensors[0]
tr_norm_c = train_ds.tensors[1]
tr_ano_s = torch.tensor(sig_tr_ano).permute(0, 2, 1).float()
tr_ano_c = torch.tensor(coe_tr_ano).float()
clf_loader = DataLoader(
    TensorDataset(torch.cat([tr_norm_s, tr_ano_s]), torch.cat([tr_norm_c, tr_ano_c]),
                  torch.zeros(len(tr_norm_s) + len(tr_ano_s))), batch_size=bs)
y_clf_tr = np.concatenate([np.zeros(len(tr_norm_s)), np.ones(len(tr_ano_s))])
print(f"train normals {len(tr_norm_s)}, train anomalies {len(tr_ano_s)}, "
      f"val beats {len(labels_val)}", flush=True)

vae_aucs, xgb_aucs = [], []
for seed in range(args.seeds):
    torch.manual_seed(seed); np.random.seed(seed)
    model = VAE1D(input_ch=1, coeff_ch=tr_norm_c.shape[1], latent_dim=ld, n_blocks=nb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(args.epochs):
        model.train()
        for xs, xc, _ in train_loader:
            xs, xc = xs.to(device), xc.to(device)
            rec, mu, logv = model(xs, xc)
            loss = nn.functional.mse_loss(rec, xs) + beta * (
                (-0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum()) / xs.size(0))
            opt.zero_grad(); loss.backward(); opt.step()

    errs_val, zs_val = compute_scores(model, val_loader, device, beta)
    errs_val = np.nan_to_num(errs_val, nan=0.0, posinf=1e6, neginf=-1e6)
    vae_auc = roc_auc_score(labels_val, -errs_val)

    errs_tr, zs_tr = compute_scores(model, clf_loader, device, beta)
    errs_tr = np.nan_to_num(errs_tr, nan=0.0, posinf=1e6, neginf=-1e6)
    clf = XGBClassifier(eval_metric="logloss")
    clf.fit(np.hstack([(-errs_tr).reshape(-1, 1), zs_tr]), y_clf_tr)
    probs = clf.predict_proba(np.hstack([(-errs_val).reshape(-1, 1), zs_val]))[:, 1]
    xgb_auc = roc_auc_score(labels_val, probs)

    vae_aucs.append(vae_auc); xgb_aucs.append(xgb_auc)
    print(f"seed {seed}: VAE AUC={vae_auc:.4f}  XGB AUC={xgb_auc:.4f}", flush=True)

x = np.array(xgb_aucs)
print("\n" + "=" * 60)
print(f"XGB AUC over {len(x)} seeds: mean={x.mean():.4f}  std={x.std(ddof=1):.4f}  "
      f"spread={x.max()-x.min():.4f}")
print("Compare that spread with the gap between the top sweep configurations:")
print("if they are the same size, the sweep ranking is dominated by noise.")
json.dump({"config": {"beta": beta, "lr": lr, "latent_dim": ld, "n_blocks": nb},
           "epochs": args.epochs, "vae_auc": vae_aucs, "xgb_auc": xgb_aucs,
           "xgb_mean": float(x.mean()), "xgb_std": float(x.std(ddof=1)),
           "xgb_spread": float(x.max() - x.min())},
          open(os.path.join(ART, "seed_noise.json"), "w"), indent=1)
print("wrote artifacts/seed_noise.json")
