import os
import pickle
import numpy as np
import pandas as pd
import uuid
from threading import Lock
from flask import Flask, render_template, request, jsonify
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.MolStandardize import charge
from rdkit.DataStructs import ConvertToNumpyArray
from tensorflow.keras.models import load_model


# Environment

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODELS_DIR = "models"
NN_DIR = "nn_data"
FP_BITS = 4096
NN_TOP_K = 10


# Load models (active + inactive for each fingerprint)

def _load(name):
    path = os.path.join(MODELS_DIR, name)
    print(f"Loading {path} ...")
    return load_model(path, compile=False)

models = {
    "ECFP4":    {"active": _load("ppb4_ecfp4_active_full_model.h5"),
                 "inactive": _load("ppb4_ecfp4_inactive_full_model.h5")},
    "AtomPair": {"active": _load("ppb4_atompair_active_full_model.h5"),
                 "inactive": _load("ppb4_atompair_inactive_full_model.h5")},
    "Layered":  {"active": _load("ppb4_layered_active_full_model.h5"),
                 "inactive": _load("ppb4_layered_inactive_full_model.h5")},
    "MAP4":     {"active": _load("ppb4_map4_active_full_model.h5"),
                 "inactive": _load("ppb4_map4_inactive_full_model.h5")},
}
print("All 8 models loaded.")

# Fingerprint types that participate in the Consensus
CONSENSUS_FPS = ("ECFP4", "AtomPair", "Layered", "MAP4")
ALL_MODEL_TYPES = list(models.keys()) + ["Consensus"]

# Load target labels

def _read_labels(filename):
    with open(os.path.join(MODELS_DIR, filename)) as f:
        return [line.strip() for line in f.readlines()[1:] if line.strip()]

active_labels   = _read_labels("PPB4_ACTIVE_DNNTARLABELS.txt")
inactive_labels = _read_labels("PPB4_INACTIVE_DNNTARLABELS.txt")
print(f"Loaded {len(active_labels):,} active labels, {len(inactive_labels):,} inactive labels.")

for fp, pair in models.items():
    a_dim = pair["active"].output_shape[-1]
    i_dim = pair["inactive"].output_shape[-1]
    assert a_dim == len(active_labels),   f"{fp} active model output ({a_dim}) != active labels ({len(active_labels)})"
    assert i_dim == len(inactive_labels), f"{fp} inactive model output ({i_dim}) != inactive labels ({len(inactive_labels)})"
print("Model output dims match label files.")


# Load target metadata

target_details = pd.read_csv(os.path.join(MODELS_DIR, "PPB4_TARGETSDETAILS.txt"), sep="\t")
target_class   = pd.read_csv(os.path.join(MODELS_DIR, "PPB4_TARGETCLASSIFICATION.txt"), sep="\t")

id_to_name     = target_details.set_index("CHEMBL_ID")["PREF_NAME"].to_dict()
id_to_class    = target_class.set_index("CHEMBL_ID")["CLASS"].to_dict()
id_to_organism = target_class.set_index("CHEMBL_ID")["ORGANISM"].to_dict()
id_to_type     = target_class.set_index("CHEMBL_ID")["TYPE"].to_dict()
print(f"Loaded metadata for {len(id_to_name):,} targets.")



# Load shared NN search data (meta + target indexes — same for all fingerprints)

print("Loading NN search metadata...")

nn_meta = {}
nn_target_index = {}
for source in ("active", "inactive"):
    nn_meta[source] = pd.read_parquet(os.path.join(NN_DIR, f"{source}_meta.parquet"))
    with open(os.path.join(NN_DIR, f"{source}_target_index.pkl"), "rb") as f:
        nn_target_index[source] = pickle.load(f)
    print(f"  {source}: {len(nn_meta[source]):,} interactions, "
          f"{len(nn_target_index[source]):,} unique targets")

# Per-fingerprint matrices and per-source fp_indices.
# All four fingerprints are loaded lazily — only the first NN query for that
# fingerprint pays the load cost; subsequent queries are instant.
NN_FP_REGISTRY = {
    "ECFP4": {
        "matrix_path": os.path.join(NN_DIR, "ecfp4_packed.npy"),
        "fp_idx_paths": {
            "active":   os.path.join(NN_DIR, "active_fp_indices.npy"),
            "inactive": os.path.join(NN_DIR, "inactive_fp_indices.npy"),
        },
        "fp_func": "ecfp4",
    },
    "AtomPair": {
        "matrix_path": os.path.join(NN_DIR, "atompair_packed.npy"),
        "fp_idx_paths": {
            "active":   os.path.join(NN_DIR, "active_atompair_fp_indices.npy"),
            "inactive": os.path.join(NN_DIR, "inactive_atompair_fp_indices.npy"),
        },
        "fp_func": "atompair",
    },
    "Layered": {
        "matrix_path": os.path.join(NN_DIR, "layered_packed.npy"),
        "fp_idx_paths": {
            "active":   os.path.join(NN_DIR, "active_layered_fp_indices.npy"),
            "inactive": os.path.join(NN_DIR, "inactive_layered_fp_indices.npy"),
        },
        "fp_func": "layered",
    },
    "MAP4": {
        "matrix_path": os.path.join(NN_DIR, "map4_packed.npy"),
        "fp_idx_paths": {
            "active":   os.path.join(NN_DIR, "active_map4_fp_indices.npy"),
            "inactive": os.path.join(NN_DIR, "inactive_map4_fp_indices.npy"),
        },
        "fp_func": "map4",
    },
}

# Per-fingerprint cache, populated on first use
_nn_data_cache = {}    # fp_name -> {"fps": matrix, "popcounts": array, "fp_indices": {source: array}}
_nn_cache_lock = Lock()

def _load_nn_data(fp_name):
    """Lazy-load the NN data for one fingerprint, or return the cached version."""
    if fp_name in _nn_data_cache:
        return _nn_data_cache[fp_name]

    with _nn_cache_lock:
        # Double-check after acquiring the lock
        if fp_name in _nn_data_cache:
            return _nn_data_cache[fp_name]

        if fp_name not in NN_FP_REGISTRY:
            raise ValueError(f"NN search not configured for fingerprint: {fp_name}")

        spec = NN_FP_REGISTRY[fp_name]
        print(f"  Lazy-loading {fp_name} NN data...")
        fps = np.load(spec["matrix_path"])
        popcounts = np.unpackbits(fps, axis=1).sum(axis=1, dtype=np.int32)
        fp_indices = {
            source: np.load(path)
            for source, path in spec["fp_idx_paths"].items()
        }
        print(f"    {fp_name} matrix: {fps.shape}, {fps.nbytes/1e6:.0f} MB")

        _nn_data_cache[fp_name] = {
            "fps": fps,
            "popcounts": popcounts,
            "fp_indices": fp_indices,
        }
        return _nn_data_cache[fp_name]


# Fingerprint calculation for predictions

def _calc_ecfp4(mol):
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS))

def _calc_atompair(mol):
    return np.array(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=FP_BITS))

def _calc_layered(mol):
    return np.array(
        Chem.LayeredFingerprint(mol, minPath=1, maxPath=7, fpSize=FP_BITS),
        dtype=np.int8,
    )

_map4_calc = None
def _get_map4_calc():
    """Lazy-init the MAP4 calculator (used by both predictions and NN queries)."""
    global _map4_calc
    if _map4_calc is None:
        from map4 import MAP4Calculator
        _map4_calc = MAP4Calculator(
            dimensions=FP_BITS, radius=2,
            is_counted=False, is_folded=True,
        )
    return _map4_calc

def _calc_map4(mol):
    return np.array(_get_map4_calc().calculate(mol))

FP_FUNCS = {
    "ECFP4":    _calc_ecfp4,
    "AtomPair": _calc_atompair,
    "Layered":  _calc_layered,
    "MAP4":     _calc_map4,
}

# Query fingerprint computation for NN search 
def _query_fp_packed(smi, fp_func_name):
    """Compute a packed-bit query fingerprint for NN search.
    fp_func_name must match one of the fp_func values in NN_FP_REGISTRY."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    if fp_func_name == "ecfp4":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
        arr = np.zeros((FP_BITS,), dtype=np.uint8)
        ConvertToNumpyArray(fp, arr)
        return np.packbits(arr)

    elif fp_func_name == "atompair":
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=FP_BITS)
        arr = np.zeros((FP_BITS,), dtype=np.uint8)
        ConvertToNumpyArray(fp, arr)
        return np.packbits(arr)

    elif fp_func_name == "layered":
        fp = Chem.LayeredFingerprint(mol, minPath=1, maxPath=7, fpSize=FP_BITS)
        arr = np.zeros((FP_BITS,), dtype=np.uint8)
        ConvertToNumpyArray(fp, arr)
        return np.packbits(arr)

    elif fp_func_name == "map4":
        # MAP4 isn't an RDKit BitVect — convert the integer array directly
        try:
            fp = _get_map4_calc().calculate(mol).astype(np.int8)
        except Exception:
            return None
        arr = (fp != 0).astype(np.uint8)
        return np.packbits(arr)

    return None


# SMILES preprocessing

_uncharger = charge.Uncharger()

def preprocess_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    largest = max(frags, key=lambda m: m.GetNumAtoms())
    cleaned = Chem.MolToSmiles(largest, isomericSmiles=False)
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return None
    mol = _uncharger.uncharge(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


# Tanimoto NN search

def _tanimoto_packed_batch(query_packed, candidate_packed_rows, candidate_popcounts):
    inter_packed = candidate_packed_rows & query_packed
    inter_count = np.unpackbits(inter_packed, axis=1).sum(axis=1, dtype=np.int32)
    query_count = int(np.unpackbits(query_packed).sum())
    union_count = candidate_popcounts + query_count - inter_count
    sim = np.zeros_like(union_count, dtype=np.float32)
    nonzero = union_count > 0
    sim[nonzero] = inter_count[nonzero] / union_count[nonzero]
    return sim


# Single-fingerprint predictions 

def _predict_single_fp(mol, fp_name, source):
    """Run one model and return a (N_targets,) numpy array of predictions."""
    fp = FP_FUNCS[fp_name](mol)
    X = fp.reshape(1, -1).astype(np.float32)
    return models[fp_name][source].predict(X, verbose=0)[0]


# Consensus: max P() across the 4 fingerprints, per target

def _consensus_predictions(mol, source):
    """
    Run all 4 fingerprint models for a given source ('active' or 'inactive')
    and return:
        - merged: (N_targets,) max prediction across models
        - winners: list of fingerprint names — winners[i] is the fp that produced merged[i]
    """
    stacked = np.stack(
        [_predict_single_fp(mol, fp, source) for fp in CONSENSUS_FPS],
        axis=0,
    )  # shape: (4, N_targets)

    merged = stacked.max(axis=0)
    winner_idx = stacked.argmax(axis=0)
    winners = [CONSENSUS_FPS[i] for i in winner_idx]
    return merged, winners


# Predict for a single SMILES 
def predict_one(smi, fp_name, num_predictions=20, mode="both"):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    out = {}

    for source, labels in (("active", active_labels), ("inactive", inactive_labels)):
        if mode not in (source, "both"):
            continue

        if fp_name == "Consensus":
            preds, winners = _consensus_predictions(mol, source)
        else:
            preds = _predict_single_fp(mol, fp_name, source)
            winners = None

        top_idx = preds.argsort()[-num_predictions:][::-1]
        out[source] = [
            _format_row(rank, idx, preds, labels,
                        source_model=(winners[idx] if winners else fp_name))
            for rank, idx in enumerate(top_idx, start=1)
        ]

    return out

def _format_row(rank, idx, preds, labels, source_model=None):
    tid = labels[idx]
    row = {
        "rank": rank,
        "target_id":   tid,
        "target_name": id_to_name.get(tid, "Unknown"),
        "confidence":  round(float(preds[idx]), 4),
        "class":       id_to_class.get(tid, "Unknown"),
        "type":        id_to_type.get(tid, "Unknown"),
        "organism":    id_to_organism.get(tid, "Unknown"),
        "url":         f"https://www.ebi.ac.uk/chembl/target_report_card/{tid}",
    }
    if source_model is not None:
        row["source_model"] = source_model
    return row


_results_cache = {}
_results_lock = Lock()
MAX_CACHED_RESULTS = 500

def _store_result(payload, result_data):
    rid = uuid.uuid4().hex[:10]
    with _results_lock:
        if len(_results_cache) >= MAX_CACHED_RESULTS:
            for k in list(_results_cache.keys())[:50]:
                _results_cache.pop(k, None)
        _results_cache[rid] = {"payload": payload, "data": result_data}
    return rid

def _get_result(rid):
    with _results_lock:
        return _results_cache.get(rid)

# Flask app

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/tutorial")
def tutorial():
    return render_template("tutorial.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/result/<rid>")
def result(rid):
    return render_template("result.html", result_id=rid)

@app.route("/api/result/<rid>")
def get_result(rid):
    entry = _get_result(rid)
    if entry is None:
        return jsonify({"error": "Result not found or expired."}), 404
    return jsonify(entry)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        smiles_list     = data.get("smiles", [])
        model_type      = data.get("model_type", "ECFP4")
        num_predictions = int(data.get("num_predictions", 20))
        mode            = data.get("mode", "both")

        if not smiles_list:
            return jsonify({"error": "No SMILES provided."}), 400
        if model_type not in ALL_MODEL_TYPES:
            return jsonify({"error": f"Unknown model_type: {model_type}"}), 400
        if mode not in ("active", "inactive", "both"):
            return jsonify({"error": f"Unknown mode: {mode}"}), 400

        results, invalid = [], []
        for raw in smiles_list:
            cleaned = preprocess_smiles(raw)
            if cleaned is None:
                invalid.append(raw)
                continue
            pred = predict_one(cleaned, model_type, num_predictions=num_predictions, mode=mode)
            if pred is None:
                invalid.append(raw)
                continue
            entry = {"smiles": raw, "canonical_smiles": cleaned}
            entry.update(pred)
            results.append(entry)

        result_data = {"results": results, "invalid_smiles": invalid}
        rid = _store_result(data, result_data)
        return jsonify({"result_id": rid})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/nearest_neighbors", methods=["POST"])
def nearest_neighbors():
    """
    Expects JSON:
        {
          "smiles":     "...",
          "target_id":  "CHEMBL204",
          "source":     "active" | "inactive",
          "model_type": "ECFP4" | "AtomPair" | "Layered" | "MAP4" | "Consensus"
        }

    For Consensus and any unconfigured model_type, falls back to ECFP4 NN search.
    """
    try:
        data = request.get_json(force=True)
        raw_smi    = (data.get("smiles") or "").strip()
        target_id  = (data.get("target_id") or "").strip()
        source     = (data.get("source") or "active").strip()
        model_type = (data.get("model_type") or "ECFP4").strip()

        if not raw_smi:
            return jsonify({"error": "No SMILES provided."}), 400
        if source not in ("active", "inactive"):
            return jsonify({"error": f"Unknown source: {source}"}), 400

        # Consensus and any unconfigured model fall back to ECFP4 NN search
        nn_fp_name = model_type if model_type in NN_FP_REGISTRY else "ECFP4"

        cleaned = preprocess_smiles(raw_smi)
        if cleaned is None:
            return jsonify({"error": f"Invalid SMILES: {raw_smi}"}), 400

        # Load (or get cached) data for this fingerprint
        nn_data = _load_nn_data(nn_fp_name)
        nn_fps = nn_data["fps"]
        nn_popcounts = nn_data["popcounts"]
        nn_fp_indices = nn_data["fp_indices"]

        # Compute the query fingerprint matching this NN matrix
        fp_func_name = NN_FP_REGISTRY[nn_fp_name]["fp_func"]
        qfp = _query_fp_packed(cleaned, fp_func_name)
        if qfp is None:
            return jsonify({"error": "Failed to compute query fingerprint."}), 400

        # Look up which meta rows belong to this target
        meta_indices = nn_target_index[source].get(target_id)
        if meta_indices is None or len(meta_indices) == 0:
            return jsonify({"neighbors": [], "nn_model": nn_fp_name})

        # Translate meta rows -> matrix rows; drop entries with no fingerprint (-1)
        fp_rows = nn_fp_indices[source][meta_indices]
        valid_mask = fp_rows >= 0
        if not valid_mask.any():
            return jsonify({"neighbors": [], "nn_model": nn_fp_name})

        meta_indices = meta_indices[valid_mask]
        fp_rows = fp_rows[valid_mask]

        candidate_fps = nn_fps[fp_rows]
        candidate_popcounts = nn_popcounts[fp_rows]

        sims = _tanimoto_packed_batch(qfp, candidate_fps, candidate_popcounts)

        if len(sims) <= NN_TOP_K:
            top_local = np.argsort(-sims)
        else:
            top_local = np.argpartition(-sims, NN_TOP_K)[:NN_TOP_K]
            top_local = top_local[np.argsort(-sims[top_local])]

        meta_df = nn_meta[source]
        neighbors = []
        for li in top_local:
            row = meta_df.iloc[int(meta_indices[li])]
            neighbors.append({
                "compound_id": row["compound_id"],
                "smiles": row["smiles"],
                "target_id": row["target_id"],
                "similarity": round(float(sims[li]), 4),
                "chembl_url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{row['compound_id']}",
            })

        return jsonify({"neighbors": neighbors, "nn_model": nn_fp_name})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "localhost")
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug, host=host, port=5000)
