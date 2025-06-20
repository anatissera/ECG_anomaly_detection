#!/usr/bin/env python3
"""Build the processed dataset from the raw FMM datasets.

Downloads PTB-XL FMM and Chapman-Shaoxing FMM from Google Drive (via the loaders)
if they are not present, concatenates them, preprocesses and saves
data/data_processed/data_processed.npz.

Two differences vs. the original notebook, both deliberate:
  * extract_lead_coeffs is fixed, so all 21 FMM coefficients are populated.
  * the TEST split is built from the real test data. The notebook passed
    `raw_dev` to build `test_proc`, which made "test" a copy of dev.
    We keep both so results stay comparable:
      X_test_*      -> real held-out test
      X_devastest_* -> dev reused as test (reproduces the original behaviour)
"""
import os, sys, json
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.dataset_loaders import get_ptb_xl_fmm_dataset, get_shaoxing_fmm_dataset
from src.preprocess_data import preprocess_data_fmm

DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.join(DATA_DIR, "data_processed")
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN, NUM_LEADS, LEAD, NUM_WAVES = 2048, 1, 0, 5
FS, BATCH = 100, 32

print("Loading FMM-enhanced PTB-XL dataset...", flush=True)
ptb = get_ptb_xl_fmm_dataset(datapath=DATA_DIR, num_leads=NUM_LEADS, lead=LEAD,
                             num_waves=NUM_WAVES, sequence_length=SEQ_LEN,
                             delete_high_A=False)
print("Loading FMM-enhanced Shaoxing dataset...", flush=True)
shx = get_shaoxing_fmm_dataset(datapath=DATA_DIR, num_leads=NUM_LEADS, lead=LEAD,
                               num_waves=NUM_WAVES, sequence_length=SEQ_LEN,
                               delete_high_A=False, split_seed=42)


# Record ids are per-folder, so give each source its own id space. PTB train and
# test come from separate official (patient-stratified) folders, hence different
# offsets; Chapman train/test are split out of one pool and must share a space so
# the no-overlap check below is meaningful.
REC_OFFSET = {('train', 'ptb'): 0, ('test', 'ptb'): 2_000_000,
              ('train', 'shx'): 1_000_000, ('test', 'shx'): 1_000_000}


def join(split_ptb, split_shx, ang_key_ptb, ang_key_shx):
    return {
        'data':             np.concatenate([ptb[split_ptb]['data'],         shx[split_shx]['data']], axis=0),
        'labels':           np.concatenate([ptb[split_ptb]['labels'],       shx[split_shx]['labels']], axis=0),
        'sizes':            np.concatenate([ptb[split_ptb]['sizes'],        shx[split_shx]['sizes']], axis=0),
        'coefficients':     np.concatenate([ptb[split_ptb]['coefficients'], shx[split_shx]['coefficients']], axis=0),
        'coefficients_ang': np.concatenate([ptb[split_ptb][ang_key_ptb],    shx[split_shx][ang_key_shx]], axis=0),
        'records':          np.concatenate([ptb[split_ptb]['records'] + REC_OFFSET[(split_ptb, 'ptb')],
                                            shx[split_shx]['records'] + REC_OFFSET[(split_shx, 'shx')]], axis=0),
    }


raw_dev  = join('train', 'train', 'ang', 'coefficients_ang')
raw_test = join('test',  'test',  'ang', 'coefficients_ang')
params = ptb['params']

dev_proc  = preprocess_data_fmm(raw_dev,  dataset_params=params, fs=FS, batch_size=BATCH, split_ecg=False)
test_proc = preprocess_data_fmm(raw_test, dataset_params=params, fs=FS, batch_size=BATCH, split_ecg=False)

X_dev_signal_raw = dev_proc['data'].astype(np.float32)
X_dev_coeffs_raw = dev_proc['coefficients'].astype(np.float32)
y_dev = dev_proc['labels'].astype(int)
records_dev = dev_proc['records'].astype(np.int64)

X_test_signal_raw = test_proc['data'].astype(np.float32)
X_test_coeffs_raw = test_proc['coefficients'].astype(np.float32)
y_test = test_proc['labels'].astype(int)
records_test = test_proc['records'].astype(np.int64)

# no recording may appear in both dev and test
overlap = set(records_dev.tolist()) & set(records_test.tolist())
assert not overlap, f"{len(overlap)} recordings leak between dev and test"

normal_class_id = params['normal_class']

# normalization stats from the NORMAL dev samples only (no leakage)
norm_mask = (y_dev == normal_class_id)
mean_x = X_dev_signal_raw[norm_mask].mean()
std_x  = X_dev_signal_raw[norm_mask].std()
mean_c = X_dev_coeffs_raw[norm_mask].mean(axis=0)
std_c  = X_dev_coeffs_raw[norm_mask].std(axis=0)

nz = int((np.abs(X_dev_coeffs_raw) > 1e-9).any(axis=0).sum())
print(f"dev {X_dev_signal_raw.shape}  test {X_test_signal_raw.shape}")
print(f"coefficient columns with any non-zero value: {nz}/21   <-- 5/21 meant the bug was active")
print(f"recordings: dev {len(np.unique(records_dev))}, test {len(np.unique(records_test))}, "
      f"beats per recording ~{len(records_dev)/max(len(np.unique(records_dev)),1):.1f}")

np.savez_compressed(
    os.path.join(OUT_DIR, "data_processed.npz"),
    X_dev_signal_raw=X_dev_signal_raw, X_dev_coeffs_raw=X_dev_coeffs_raw, y_dev=y_dev,
    X_test_signal_raw=X_test_signal_raw, X_test_coeffs_raw=X_test_coeffs_raw, y_test=y_test,
    records_dev=records_dev, records_test=records_test,
    mean_x=mean_x, std_x=std_x, mean_c=mean_c, std_c=std_c,
    normal_class_id=normal_class_id,
)
print("saved", os.path.join(OUT_DIR, "data_processed.npz"))
