"""
Rebuild only the active/inactive metadata parquets with real ChEMBL compound IDs.
Reuses the existing fingerprint matrix and target indexes.
"""
import os
import pandas as pd
import numpy as np
import pickle

OUTPUT_DIR = "nn_data"

# Load the updated filtered_df
print("Loading filtered_df.parquet ...")
filtered_df = pd.read_parquet("filtered_df.parquet")
filtered_df = filtered_df.reset_index(drop=True)
print(f"  Loaded {len(filtered_df):,} rows")
print(f"  Columns: {filtered_df.columns.tolist()}")

if "compound_chembl_id" not in filtered_df.columns:
    raise SystemExit("ERROR: filtered_df does not have a 'compound_chembl_id' column. Add it and re-save the parquet first.")

# Filter to compounds whose fingerprint exists in the cache
smiles_arr = np.load(os.path.join(OUTPUT_DIR, "ecfp4_smiles.npy"), allow_pickle=True)
valid_smiles = set(smiles_arr.tolist())
filtered_df = filtered_df[filtered_df["nonisomeric_smiles"].isin(valid_smiles)].reset_index(drop=True)
print(f"  After filtering for valid fingerprints: {len(filtered_df):,} rows")

def rebuild_meta(activity_label, basename):
    df = filtered_df[filtered_df["activity"] == activity_label].reset_index(drop=True)
    print(f"\n=== Rebuilding {basename}_meta.parquet: {len(df):,} rows ===")

    meta = pd.DataFrame({
        "compound_id": df["compound_chembl_id"].astype(str),
        "smiles":      df["nonisomeric_smiles"].astype(str),
        "target_id":   df["target_chembl_id"].astype(str),
    })

    # Sanity check: make sure the row count matches the existing fp_indices file
    fp_idx = np.load(os.path.join(OUTPUT_DIR, f"{basename}_fp_indices.npy"))
    if len(meta) != len(fp_idx):
        raise SystemExit(
            f"Row count mismatch! meta={len(meta)} vs existing fp_indices={len(fp_idx)}.\n"
            f"This means filtered_df has changed beyond just adding the compound_chembl_id column.\n"
            f"You'll need to do a full rebuild with build_nn_data.py instead."
        )

    meta_path = os.path.join(OUTPUT_DIR, f"{basename}_meta.parquet")
    meta.to_parquet(meta_path)
    print(f"  Saved {meta_path}")

    # Show a sample so you can verify
    print(f"  Sample rows:")
    print(meta.head(3).to_string(index=False))

rebuild_meta("active",   "active")
rebuild_meta("inactive", "inactive")
print("\nDone.")