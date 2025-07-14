import numpy as np
import torch
from torch.utils.data import TensorDataset


def split_train_val_with_raw_dev(
    dev_norm_signal_raw, dev_norm_coeffs_raw,
    dev_ano_signal_raw,  dev_ano_coeffs_raw,
    val_frac=0.2, seed=42
):
    # dividir normales en train y val
    N = len(dev_norm_signal_raw)
    vsize = int(val_frac * N)
    train_size = N - vsize

    idx = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    idx_tr, idx_vn = idx[:train_size], idx[train_size:]

    sig_tr_raw = dev_norm_signal_raw[idx_tr]
    coe_tr_raw = dev_norm_coeffs_raw[idx_tr]
    sig_vn_raw = dev_norm_signal_raw[idx_vn]
    coe_vn_raw = dev_norm_coeffs_raw[idx_vn]

    # media/std sólo sobre train_norm
    mean_sig = sig_tr_raw.mean()
    std_sig  = sig_tr_raw.std() + 1e-8
    mean_coe = coe_tr_raw.mean(axis=0)
    std_coe  = coe_tr_raw.std(axis=0) + 1e-8

    # normalizar
    sig_tr = (sig_tr_raw - mean_sig) / std_sig
    coe_tr = (coe_tr_raw - mean_coe) / std_coe

    sig_vn = (sig_vn_raw - mean_sig) / std_sig
    coe_vn = (coe_vn_raw - mean_coe) / std_coe

    # muestreae anomalías para val
    idx_a = rng.choice(len(dev_ano_signal_raw), size=vsize, replace=False)
    sig_va_raw = dev_ano_signal_raw[idx_a]
    coe_va_raw = dev_ano_coeffs_raw[idx_a]

    sig_va = (sig_va_raw - mean_sig) / std_sig
    coe_va = (coe_va_raw - mean_coe) / std_coe

    train_ds = TensorDataset(
        torch.tensor(sig_tr).permute(0,2,1),
        torch.tensor(coe_tr),
        torch.zeros(train_size)
    )

    sig_val = np.concatenate([sig_vn, sig_va], axis=0)
    coe_val = np.concatenate([coe_vn, coe_va], axis=0)
    lbl_val = np.concatenate([np.zeros(len(sig_vn)), np.ones(len(sig_va))])

    val_ds = TensorDataset(
        torch.tensor(sig_val).permute(0,2,1),
        torch.tensor(coe_val),
        torch.tensor(lbl_val)
    )

    stats = {
        'mean_sig': mean_sig, 'std_sig': std_sig,
        'mean_coe': mean_coe, 'std_coe': std_coe
    }
    return train_ds, val_ds, stats, sig_tr, coe_tr, sig_vn, coe_vn, sig_va, coe_va, idx_a