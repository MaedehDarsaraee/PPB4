# PPB4 : Polypharmacology Browser 4

Target prediction from a chemical structure, using multitask deep neural
networks trained on ChEMBL 35. PPB4 predicts both the targets a molecule is
likely to **hit** and those it is likely to be **inactive** against.

Public instance: <https://ppb4.gdb.tools>

---

## What this repository contains

Only the application code. The trained models and the nearest-neighbour search
data are ~4.8 GB and are distributed separately (see below).

```
ppb4_predict.py        Flask app: prediction, nearest-neighbour search, result pages
templates/             Home, results, tutorial, FAQ, contact
static/                JSME structure editor and assets
Dockerfile             Container definition
docker-compose.yml     Runs the container with the data mounted
requirements.txt       Python dependencies
build_*.py             Scripts used to build nn_data/ (not needed to run PPB4)
```

## Data you need to download

Two directories, both mounted into the container at run time.

### `models/`

Eight Keras networks (~95 MB each) plus four small lookup files:

| file | purpose |
|---|---|
| `ppb4_{ecfp4,atompair,layered,map4}_{active,inactive}_full_model.h5` | the eight DNNs |
| `PPB4_ACTIVE_DNNTARLABELS.txt` | 7,552 target IDs, active model output order |
| `PPB4_INACTIVE_DNNTARLABELS.txt` | 7,178 target IDs, inactive model output order |
| `PPB4_TARGETSDETAILS.txt` | target names, types, organisms |
| `PPB4_TARGETCLASSIFICATION.txt` | protein class annotations |

All eight models are loaded at start-up, so all eight must be present.

### `nn_data/` 

Powers the "similar known compounds" panel, which shows the training compounds
most similar to your query for a given target. **Predictions work without it**
once the code change noted under *Running without neighbour search* is applied.

It splits into two tiers:

| tier | files | size | when it loads |
|---|---|---|---|
| metadata | `{active,inactive}_meta.parquet`, `{active,inactive}_target_index.pkl` | 125 MB | at start-up |
| per fingerprint | `<fp>_packed.npy`, `{active,inactive}_<fp>_fp_indices.npy` | 885 MB each | first neighbour query for that fingerprint |

You only need the per-fingerprint block for fingerprints you actually want to
search with. Note that **Consensus predictions fall back to ECFP4** for the
neighbour search, so `ecfp4_packed.npy` covers the recommended model.

The `*_smiles.npy` files are build artefacts of `build_nn_indexes.py` and are
not read at run time — they do not need to be downloaded.

## Requirements

| | |
|---|---|
| RAM | 8 GB minimum. Add ~1 GB for each fingerprint you run neighbour searches with. |
| Disk | 0.9 GB for predictions only, up to 4.8 GB with all four neighbour-search fingerprints. |
| CPU | Any x86-64. There is no GPU requirement; inference is a single forward pass. |
| Python | 3.9 (the pinned TensorFlow 2.10 build does not support newer versions). |

## Running with Docker

```bash
docker build -t ppb4 .

docker run -p 5000:5000 \
  -v /path/to/models:/app/models:ro \
  -v /path/to/nn_data:/app/nn_data:ro \
  ppb4
```

or, with the paths set in `docker-compose.yml`:

```bash
docker compose up
```

Then open <http://localhost:5000>.

## Running without Docker

```bash
pip install -r requirements.txt
pip install tmap-viz
pip install git+https://github.com/reymond-group/map4.git

FLASK_HOST=0.0.0.0 FLASK_DEBUG=false python ppb4_predict.py
```

`models/` and `nn_data/` must sit next to `ppb4_predict.py`.

## Running without neighbour search

`ppb4_predict.py` reads the four `nn_data/` metadata files at import, so the app
will not start if `nn_data/` is absent. To run predictions only, wrap that block
(the `for source in ("active", "inactive")` loop that fills `nn_meta` and
`nn_target_index`) in a `try`/`except` and disable the `/nearest_neighbors`
route. Total footprint then drops to about 890 MB.

## Notes

- **First start is slow.** Eight Keras models and 125 MB of metadata load before
  the first request is served. Allow several minutes and at least 8 GB of RAM.
- **Results are held in memory**, not on disk. A restart invalidates every
  `/result/<id>` URL. The `queries/` and `results/` directories are unused.
- **MAP4 is installed from git at build time.** If that repository moves or
  changes, the image will no longer build. Pin it to a commit for anything you
  need to reproduce later.

## Data availability

`models/` and `nn_data/` are hosted separately because of their size:

> **Download:** _link to be added_

## License

_To be decided before public release._

The bundled structure editor in `static/jsme/` is **JSME**, by Peter Ertl and
Bruno Bienfait, and is redistributed under its own terms. See
<https://jsme-editor.github.io/>.

## Contact

Questions and bug reports: <https://ppb4.gdb.tools/contact>

## Citing

- **PPB4 preprint:** in preparation
- **PPB3 paper:** *J. Chem. Inf. Model.* 2026, 66, 2466–2473 —
  <https://doi.org/10.1021/acs.jcim.6c00299>

Reymond Group, Department of Chemistry, Biochemistry and Pharmaceutical
Sciences, University of Bern.
