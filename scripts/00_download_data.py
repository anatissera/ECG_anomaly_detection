#!/usr/bin/env python3
"""Download the raw FMM datasets into data/.

The datasets are not committed to the repository (they are several GB). The
loaders fetch the FMM-preprocessed PTB-XL and Chapman-Shaoxing datasets from
Google Drive on first use; this script just triggers that download so the data
is in place before running the pipeline.

After this, run scripts/01_build_dataset.py to produce data_processed.npz.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.dataset_loaders import get_ptb_xl_fmm_dataset, get_shaoxing_fmm_dataset

DATA = os.path.join(REPO, "data")
os.makedirs(DATA, exist_ok=True)

print("Downloading FMM-preprocessed PTB-XL dataset...")
get_ptb_xl_fmm_dataset(datapath=DATA, num_leads=1, lead=0, num_waves=5,
                       sequence_length=2048, delete_high_A=False)

print("Downloading FMM-preprocessed Chapman-Shaoxing dataset...")
get_shaoxing_fmm_dataset(datapath=DATA, num_leads=1, lead=0, num_waves=5,
                         sequence_length=2048, delete_high_A=False, split_seed=42)

print("Done. Raw datasets are in data/. Next: python scripts/01_build_dataset.py")
