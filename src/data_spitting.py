import numpy as np
import torch
from torch.utils.data import TensorDataset


def split_train_val_with_raw_dev(
    dev_norm_signal_raw, dev_norm_coeffs_raw, dev_norm_records,
    dev_ano_signal_raw,  dev_ano_coeffs_raw,  dev_ano_records,
    val_frac=0.2, seed=42
):
    """Split the dev set into train/validation, grouped by recording.

    Every ECG recording contributes several beats. The split is done over the
    set of recordings, not over individual beats, so no recording has beats on
    both sides. Splitting beats at random lets the model memorise a patient in
    training and then be scored on the same patient in validation, which makes
    the validation metrics far too optimistic.

    Normalisation statistics come from the training normals only.
    """
    rng = np.random.RandomState(seed)

    # ---- split the RECORDINGS, not the beats
    all_records = np.unique(np.concatenate([dev_norm_records, dev_ano_records]))
    rng.shuffle(all_records)
    n_val_rec = int(round(val_frac * len(all_records)))
    val_records = set(all_records[:n_val_rec].tolist())

    norm_is_val = np.array([r in val_records for r in dev_norm_records])
    ano_is_val = np.array([r in val_records for r in dev_ano_records])

    idx_tr = np.where(~norm_is_val)[0]
    idx_vn = np.where(norm_is_val)[0]
    idx_val_ano = np.where(ano_is_val)[0]

    sig_tr_raw = dev_norm_signal_raw[idx_tr]
    coe_tr_raw = dev_norm_coeffs_raw[idx_tr]
    sig_vn_raw = dev_norm_signal_raw[idx_vn]
    coe_vn_raw = dev_norm_coeffs_raw[idx_vn]

    # mean/std computed on train_norm only
    mean_sig = sig_tr_raw.mean()
    std_sig = sig_tr_raw.std() + 1e-8
    mean_coe = coe_tr_raw.mean(axis=0)
    std_coe = coe_tr_raw.std(axis=0) + 1e-8

    # normalize
    sig_tr = (sig_tr_raw - mean_sig) / std_sig
    coe_tr = (coe_tr_raw - mean_coe) / std_coe

    sig_vn = (sig_vn_raw - mean_sig) / std_sig
    coe_vn = (coe_vn_raw - mean_coe) / std_coe

    # validation anomalies: those belonging to validation recordings
    sig_va = (dev_ano_signal_raw[idx_val_ano] - mean_sig) / std_sig
    coe_va = (dev_ano_coeffs_raw[idx_val_ano] - mean_coe) / std_coe

    train_ds = TensorDataset(
        torch.tensor(sig_tr).permute(0, 2, 1),
        torch.tensor(coe_tr),
        torch.zeros(len(sig_tr))
    )

    sig_val = np.concatenate([sig_vn, sig_va], axis=0)
    coe_val = np.concatenate([coe_vn, coe_va], axis=0)
    lbl_val = np.concatenate([np.zeros(len(sig_vn)), np.ones(len(sig_va))])

    val_ds = TensorDataset(
        torch.tensor(sig_val).permute(0, 2, 1),
        torch.tensor(coe_val),
        torch.tensor(lbl_val)
    )

    stats = {
        'mean_sig': mean_sig, 'std_sig': std_sig,
        'mean_coe': mean_coe, 'std_coe': std_coe
    }
    return train_ds, val_ds, stats, sig_tr, coe_tr, sig_vn, coe_vn, sig_va, coe_va, idx_val_ano
