# Bodhi VLM

Privacy budget assessment for vision and vision–language models via bottom-up, top-down feature search and expectation–maximization analysis.

This repository contains the core Python code for:

- BUA/TDA-style grouping of hierarchical features,
- EMPA-style privacy budget assessment,
- Synthetic experiments comparing EMPA with Chi-square, K-L divergence, and MMD,
- Example integrations with YOLO and CLIP.

All public files in this repository are code and documentation only.

## Repository layout

- `src/`
  - Core experiment code (`metrics.py`, `grouping.py`, `run_experiments.py`)
  - High-level Bodhi VLM pipeline API (`bodhi_vlm_pipeline.py`)
  - Example integrations (`yolo_bodhi_example.py`, `clip_bodhi_example.py`)
- `results/`
  - Optional directory for experiment outputs (created on demand)

Other directories under the project root (if present) are internal and do not need to be exposed when sharing the source code.

## Installation

It is recommended to use Python 3.10+ and install dependencies from the root:

```bash
pip install -r requirements.txt
```

This installs the core dependencies for synthetic experiments and plotting (`numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`).

Optional dependencies for the integration examples:

```bash
pip install ultralytics
pip install git+https://github.com/openai/CLIP.git
```

## Running synthetic experiments

The synthetic experiment runner is `src/run_experiments.py`. It generates multi-layer synthetic features, adds privacy noise, runs BUA/TDA grouping and EMPA, and compares with baseline metrics.

Basic usage:

```bash
cd src
python run_experiments.py
```

This runs experiments for `epsilon = 0.1, 0.01` and writes results to `../results` by default (or to the directory specified with `--out_dir`).

More control:

```bash
cd src

# Custom output directory and multiple privacy budgets
python run_experiments.py --out_dir ../results --epsilon 0.1 0.01 0.001

# Larger number of samples and layers (closer to VLM-style hierarchies)
python run_experiments.py --n_samples 500 --n_layers 6 --epsilon 0.1 0.01
```

The script writes:

- `bodhi_vlm_metrics.csv` (per-ε metrics: chi2, kl, mmd, rmse, empa_bias_bua, empa_bias_tda)
- `bodhi_vlm_empa_bias.png`, `bodhi_vlm_metrics_vs_epsilon.png` (if `matplotlib` is available)

## Using the Bodhi VLM pipeline in your code

The high-level pipeline API is provided in `src/bodhi_vlm_pipeline.py`. Typical usage:

```python
from bodhi_vlm_pipeline import assess_privacy_budget_from_features

# layer_features_orig / layer_features_noised: list of arrays, each (N_i, D)
# sensitive_per_layer: list of boolean masks, each length N_i
metrics = assess_privacy_budget_from_features(
    layer_features_orig,
    layer_features_noised,
    sensitive_per_layer,
    epsilon=0.1,
)
print(metrics)
```

This returns a dictionary with:

- `epsilon`
- baseline metrics: `chi2`, `kl`, `mmd`, `rmse`
- EMPA-style metrics: `empa_bias_bua`, `empa_bias_tda`

You can plug in features from any backbone or VLM that exposes layer-wise feature tensors.

## Integration examples (YOLO / CLIP)

Two example scripts demonstrate how to integrate Bodhi VLM with real models:

- `src/yolo_bodhi_example.py`:
  - Uses `ultralytics.YOLO` (e.g., `yolov8n.pt`)
  - Hooks into selected backbone layers to collect feature maps
  - Adds Gaussian noise to inputs
  - Builds simple sensitive masks and calls `assess_privacy_budget_from_features`

- `src/clip_bodhi_example.py`:
  - Uses OpenAI CLIP ViT-B/32
  - Hooks into transformer blocks in the visual encoder to collect patch features
  - Adds Laplace noise to inputs
  - Marks some images as sensitive and builds per-layer masks
  - Calls `assess_privacy_budget_from_features`

To run these examples (after installing the optional dependencies):

```bash
cd src

python yolo_bodhi_example.py   # YOLO integration example
python clip_bodhi_example.py   # CLIP integration example
```

These scripts are intended as starting points; in your own system you should replace the dummy images and sensitivity logic with your actual data and privacy semantics.
