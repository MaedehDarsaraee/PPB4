"""
Build MAP4 NN search data files.

Reads filtered_df.parquet and produces:
    nn_data/
        map4_packed.npy             ← packed-bit MAP4 fingerprint matrix
        map4_smiles.npy             ← parallel SMILES array
        active_map4_fp_indices.npy  ← active meta-row -> fp matrix row
        inactive_map4_fp_indices.npy

Reuses existing active_meta.parquet, inactive_meta.parquet,
active_target_index.pkl, inactive_target_index.pkl.

Fingerprint matches training-time calculation EXACTLY:
    MAP4Calculator(dimensions=4096, radius=2, is_counted=False, is_folded=True)
    .calculate(mol).astype(np.int8)

NOTE: MAP4 is much slower than the other fingerprints. This script uses
joblib parallelism (matching your training-time approach) to bring the
total time from ~3 hours to ~30-60 minutes.

Usage:
    python build_map4_nn.py
"""

import os
import gc
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from joblib import Parallel, delayed

OUTPUT_DIR = "nn_data"
FP_PACKED_PATH = os.path.join(OUTPUT_DIR, "map4_packed.npy")
FP_SMILES_PATH = os.path.join(OUTPUT_DIR, "map4_smiles.npy")
FP_BITS = 4096
PACKED_BYTES = FP_BITS // 8
N_JOBS = 10            # number of parallel workers
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
# Compute MAP4 fingerprints in parallel (matches training-time calculation)
# ---------------------------------------------------------------------------
def _process_chunk(smiles_list):
    """Worker function: compute packed MAP4 fingerprints for one chunk.
    Imports MAP4Calculator INSIDE the worker because it isn't picklable.
    Returns a list of (smi, packed_fp) tuples for compounds that succeeded.
    """
    from map4 import MAP4Calculator
    calculator = MAP4Calculator(
        dimensions=FP_BITS, radius=2,
        is_counted=False, is_folded=True,
    )

    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            fp = calculator.calculate(mol).astype(np.int8)
            # Convert to packed bits (MAP4 with is_folded=True returns 0/1 ints)
            arr = (fp != 0).astype(np.uint8)
            packed = np.packbits(arr)
            out.append((smi, packed))
        except Exception:
            continue
    return out

def compute_fingerprints():
    print(f"\nComputing MAP4 fingerprints for {len(unique_smiles):,} compounds "
          f"in parallel ({N_JOBS} workers)...")

    # Split SMILES into N_JOBS chunks
    smiles_chunks = np.array_split(unique_smiles, N_JOBS)
    smiles_chunks = [c.tolist() for c in smiles_chunks]

    # Run workers
    chunk_results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_process_chunk)(chunk) for chunk in smiles_chunks
    )

    # Combine results
    smiles_list = []
    fp_list = []
    for chunk_out in chunk_results:
        for smi, packed in chunk_out:
            smiles_list.append(smi)
            fp_list.append(packed)

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
    print(f"\nBuilding {basename}_map4_fp_indices ...")

    meta_path = os.path.join(OUTPUT_DIR, f"{basename}_meta.parquet")
    if not os.path.exists(meta_path):
        raise SystemExit(
            f"ERROR: {meta_path} not found. Build the ECFP4 data first "
            f"(it creates the shared meta files)."
        )
    meta = pd.read_parquet(meta_path)
    print(f"  Loaded {len(meta):,} interactions from {meta_path}")

    # Map each meta row to a row in the MAP4 fingerprint matrix
    fp_indices = np.full(len(meta), -1, dtype=np.int64)
    for i, smi in enumerate(meta["smiles"]):
        idx = smi_to_idx.get(smi)
        if idx is not None:
            fp_indices[i] = idx

    n_missing = int((fp_indices == -1).sum())
    if n_missing:
        print(f"  WARNING: {n_missing:,} interactions reference SMILES with no MAP4 fingerprint")
        print(f"  These interactions will be excluded from MAP4 NN search.")

    fp_idx_path = os.path.join(OUTPUT_DIR, f"{basename}_map4_fp_indices.npy")
    np.save(fp_idx_path, fp_indices)
    print(f"  Saved {fp_idx_path}")

build_fp_indices("active",   "active")
build_fp_indices("inactive", "inactive")

print("\nDone.")
print(f"\nFiles produced:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if "map4" in f:
        full = os.path.join(OUTPUT_DIR, f)
        print(f"  {f}  ({os.path.getsize(full)/1e6:.1f} MB)")