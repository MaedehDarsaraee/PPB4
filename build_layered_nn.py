"""
Build Layered NN search data files.

Reads filtered_df.parquet and produces:
    nn_data/
        layered_packed.npy             ← packed-bit Layered fingerprint matrix
        layered_smiles.npy             ← parallel SMILES array
        active_layered_fp_indices.npy  ← active meta-row -> fp matrix row
        inactive_layered_fp_indices.npy

Reuses existing active_meta.parquet, inactive_meta.parquet,
active_target_index.pkl, inactive_target_index.pkl.

Fingerprint matches training-time calculation EXACTLY:
    Chem.LayeredFingerprint(mol, minPath=1, maxPath=7, fpSize=4096)

Usage:
    python build_layered_nn.py
"""

import os
import gc
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm import tqdm

OUTPUT_DIR = "nn_data"
FP_PACKED_PATH = os.path.join(OUTPUT_DIR, "layered_packed.npy")
FP_SMILES_PATH = os.path.join(OUTPUT_DIR, "layered_smiles.npy")
FP_BITS = 4096
PACKED_BYTES = FP_BITS // 8
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load filtered_df
# ---------------------------------------------------------------------------
print("Loading filtered_df.parquet ...")
filtered_df = pd.read_parquet("filtered_df.parquet")
filtered_df = filtered_df.reset_index(drop=True)
print(f"  Loaded {len(filtered_df):,} interactions")

unique_smiles = filtered_df["nonisomeric_smiles"].unique()
print(f"  Unique compounds: {len(unique_smiles):,}")

# ---------------------------------------------------------------------------
# Compute Layered fingerprints (matches training-time calculation exactly)
# ---------------------------------------------------------------------------
def smi_to_packed_layered(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = Chem.LayeredFingerprint(mol, minPath=1, maxPath=7, fpSize=FP_BITS)
    arr = np.zeros((FP_BITS,), dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return np.packbits(arr)

def compute_fingerprints():
    print(f"\nComputing Layered fingerprints for {len(unique_smiles):,} compounds...")
    smiles_list = []
    fp_list = []
    for smi in tqdm(unique_smiles):
        fp = smi_to_packed_layered(smi)
        if fp is not None:
            smiles_list.append(smi)
            fp_list.append(fp)
    smiles_arr = np.array(smiles_list, dtype=object)
    fps_arr = np.stack(fp_list, axis=0).astype(np.uint8)
    print(f"  Computed {len(smiles_list):,} valid fingerprints "
          f"({fps_arr.nbytes/1e6:.0f} MB)")
    return smiles_arr, fps_arr

# Skip recomputation if already built
if os.path.exists(FP_PACKED_PATH) and os.path.exists(FP_SMILES_PATH):
    print(f"  Cache exists — reloading from {OUTPUT_DIR}/ ...")
    smiles_arr = np.load(FP_SMILES_PATH, allow_pickle=True)
    fps_arr = np.load(FP_PACKED_PATH)
    print(f"  Loaded {len(smiles_arr):,} fingerprints")

    # Verify cache covers what we need
    needed = set(unique_smiles.tolist())
    cached = set(smiles_arr.tolist())
    missing = needed - cached
    if missing:
        print(f"  Cache missing {len(missing):,} SMILES — recomputing.")
        smiles_arr, fps_arr = compute_fingerprints()
        np.save(FP_SMILES_PATH, smiles_arr)
        np.save(FP_PACKED_PATH, fps_arr)
else:
    smiles_arr, fps_arr = compute_fingerprints()
    print(f"  Saving {FP_SMILES_PATH} ...")
    np.save(FP_SMILES_PATH, smiles_arr)
    print(f"  Saving {FP_PACKED_PATH} ...")
    np.save(FP_PACKED_PATH, fps_arr)

print(f"\nFingerprint matrix: shape={fps_arr.shape}, dtype={fps_arr.dtype}, "
      f"size={fps_arr.nbytes/1e6:.0f} MB")

# ---------------------------------------------------------------------------
# Build per-class fp_indices
# ---------------------------------------------------------------------------
smi_to_idx = {s: i for i, s in enumerate(smiles_arr)}

def build_fp_indices(activity_label, basename):
    print(f"\nBuilding {basename}_layered_fp_indices ...")

    meta_path = os.path.join(OUTPUT_DIR, f"{basename}_meta.parquet")
    if not os.path.exists(meta_path):
        raise SystemExit(
            f"ERROR: {meta_path} not found. Build the ECFP4 data first "
            f"(it creates the shared meta files)."
        )
    meta = pd.read_parquet(meta_path)
    print(f"  Loaded {len(meta):,} interactions from {meta_path}")

    # Map each meta row to a row in the Layered fingerprint matrix
    fp_indices = np.full(len(meta), -1, dtype=np.int64)
    for i, smi in enumerate(meta["smiles"]):
        idx = smi_to_idx.get(smi)
        if idx is not None:
            fp_indices[i] = idx

    n_missing = int((fp_indices == -1).sum())
    if n_missing:
        print(f"  WARNING: {n_missing:,} interactions reference SMILES with no Layered fingerprint")
        print(f"  These interactions will be excluded from Layered NN search.")

    fp_idx_path = os.path.join(OUTPUT_DIR, f"{basename}_layered_fp_indices.npy")
    np.save(fp_idx_path, fp_indices)
    print(f"  Saved {fp_idx_path}")

build_fp_indices("active",   "active")
build_fp_indices("inactive", "inactive")

print("\nDone.")
print(f"\nFiles produced:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if "layered" in f:
        full = os.path.join(OUTPUT_DIR, f)
        print(f"  {f}  ({os.path.getsize(full)/1e6:.1f} MB)")