# Bodhi VLM

Privacy budget assessment for vision and vision–language models via bottom-up (BUA), top-down (TDA) feature search and expectation–maximization (EMPA) analysis.

This repository contains:

- **Core Python code**: BUA/TDA grouping, EMPA-style assessment, synthetic experiments, and a high-level pipeline API.
- **Example integrations**: YOLO (ultralytics) and CLIP (OpenAI) with hooks for layer-wise features.
- **Paper submodule** (optional): LaTeX manuscript and figures in `paper/` (Overleaf-backed repo).

All instructions below are in English.

---

## Repository layout

| Path | Description |
|------|-------------|
| `paper/` | Git submodule pointing to the Overleaf manuscript repo. Populated when cloning with `--recurse-submodules` or after `git submodule update --init`. |
| `src/` | Core code: `metrics.py`, `grouping.py`, `run_experiments.py`, `bodhi_vlm_pipeline.py`, and example scripts `yolo_bodhi_example.py`, `clip_bodhi_example.py`. |
| `requirements.txt` | Python dependencies for experiments and plotting. |
| `results/` | Optional output directory for experiment CSV and figures (created on demand). |

---

## Clone (including the paper submodule)

To clone and fetch the paper submodule in one step:

```bash
git clone --recurse-submodules https://github.com/<your-org>/bodhi-vlm.git
cd bodhi-vlm
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

To pull the main repo and update submodules together:

```bash
git pull --recurse-submodules
```

Or set once so that `git pull` always updates submodules:

```bash
git config submodule.recurse true
```

---

## Installation

Use Python 3.10+ and install from the project root:

```bash
pip install -r requirements.txt
```

This installs: `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`.

Optional dependencies for the YOLO and CLIP examples:

```bash
pip install ultralytics
pip install git+https://github.com/openai/CLIP.git
```

---

## Running synthetic experiments

From the project root:

```bash
cd src
python run_experiments.py
```

Defaults: privacy budgets `epsilon = 0.1, 0.01`; output under `../results`. Override with:

```bash
python run_experiments.py --out_dir ../results --epsilon 0.1 0.01 0.001
python run_experiments.py --n_samples 500 --n_layers 6 --epsilon 0.1 0.01
```

Outputs:

- `bodhi_vlm_metrics.csv` (per-ε: chi2, kl, mmd, rmse, empa_bias_bua, empa_bias_tda)
- `bodhi_vlm_empa_bias.png`, `bodhi_vlm_metrics_vs_epsilon.png` (if matplotlib is installed)

---

## Using the pipeline API

Use the high-level API in `src/bodhi_vlm_pipeline.py`:

```python
from bodhi_vlm_pipeline import assess_privacy_budget_from_features

# layer_features_orig / layer_features_noised: list of arrays, shape (N_i, D) per layer
# sensitive_per_layer: list of boolean masks, length N_i per layer
metrics = assess_privacy_budget_from_features(
    layer_features_orig,
    layer_features_noised,
    sensitive_per_layer,
    epsilon=0.1,
)
# Returns: epsilon, chi2, kl, mmd, rmse, empa_bias_bua, empa_bias_tda
```

Works with any backbone or VLM that exposes layer-wise feature tensors.

---

## Integration examples (YOLO / CLIP)

After installing the optional dependencies:

```bash
cd src
python yolo_bodhi_example.py   # ultralytics YOLO backbone + Gaussian noise
python clip_bodhi_example.py   # OpenAI CLIP ViT-B/32 + Laplace noise
```

Both scripts hook into the model, add noise, build simple sensitive masks, and call `assess_privacy_budget_from_features`. Replace the dummy inputs and sensitivity logic with your own data and privacy definitions.

---

## License and contact

See the repository and submodule repositories for license and contact information.
