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
| `src/` | Source code: unified entry (`main.py` + `config.json`), utilities (`utils/`), core pipeline (`core/`), experiments (`experiments/`), and examples (`examples/`). |
| `requirements.txt` | Python dependencies for experiments, figures, and detector/VLM integrations. |
| `results/` | Optional output directory for experiment CSV and figures (created on demand). |

---

## Unified entry (recommended)

All experiments are controlled by **`src/config.json`** and run from a single entry point:

```bash
cd src
python main.py
```

- Edit **`src/config.json`**: set `"enabled": true` for each experiment you want to run (`synthetic`, `detector`, `vlm`, `aggregate`, `interpretability`), and adjust parameters (e.g. `epsilon`, `seeds`, `out_dir`).
- Override output directory: `python main.py --out_dir ../results`
- Run only specific experiments: `python main.py --experiments synthetic aggregate`
- Use a different config file: `python main.py --config my_config.json`

---

## Source layout

The `src/` directory is organized as follows:

| Path | Description |
|------|-------------|
| `main.py` | Single unified entrypoint that reads `config.json` and runs enabled experiments. |
| `config.json` | Controls `out_dir` and per-experiment parameters (`epsilon`, `seeds`, `num_images`, etc.). |
| `utils/metrics.py` | Metrics: Chi-square, K-L, MMD, rMSE, EMPA bias, `compare_metrics`. |
| `utils/grouping.py` | Grouping: MDAV-like clustering, BUA/TDA (`mdav_like_cluster`, `bua_style`, `tda_style`). |
| `core/pipeline.py` | High-level pipeline: `assess_privacy_budget_from_features`, `group_features_bua_tda`. |
| `experiments/synthetic.py` | Synthetic multi-seed experiments → `bodhi_vlm_metrics.csv` and summary figures. |
| `experiments/detector.py` | Detector experiments (YOLO/DETR) → `detector_metrics.csv`. |
| `experiments/vlm.py` | VLM experiments (CLIP/BLIP) → `vlm_metrics.csv`. |
| `experiments/aggregate.py` | Aggregate multi-seed metrics and emit LaTeX tables (`detector_tab_rmse.tex`, `vlm_tab_results.tex`). |
| `experiments/interpretability.py` | Interpretability figures → `bodhi_vlm_sensitive_dist.png`, `bodhi_vlm_tsne.png`. |
| `examples/` | Optional example scripts (YOLO / CLIP integrations). |
| `scripts/` | Backwards-compatible CLI wrappers that call functions in `experiments/`. |
| `metrics.py`, `grouping.py`, `bodhi_vlm_pipeline.py` | Thin compatibility shims that re-export from `utils/` and `core/`. |

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

**Recommended:** enable `synthetic` in `src/config.json` and run `python main.py` from `src/`.

Outputs:

- `results/bodhi_vlm_metrics.csv` (per-ε: chi2, kl, mmd, rmse, empa_bias_bua, empa_bias_tda)
- `results/bodhi_vlm_empa_bias.png`, `results/bodhi_vlm_metrics_vs_epsilon.png` (if matplotlib is installed)

---

## Using the pipeline API

Use the high-level API in `src/core/pipeline.py`:

```python
from core.pipeline import assess_privacy_budget_from_features

# layer_features_orig / layer_features_noised: list of arrays, shape (N_i, D) per layer
# sensitive_per_layer: list of boolean masks, length N_i per layer
metrics = assess_privacy_budget_from_features(
    layer_features_orig,
    layer_features_noised,
    sensitive_per_layer,
    epsilon=0.1,
)
# Returns: epsilon, chi2, kl, mmd, rmse, wass1 (optional),
#          empa_bias_bua, empa_bias_tda
```

Works with any backbone or VLM that exposes layer-wise feature tensors.

---

## Integration examples (YOLO / CLIP)

After installing the optional dependencies:

```bash
cd src
python examples/yolo_bodhi_example.py   # ultralytics YOLO backbone + Gaussian noise
python examples/clip_bodhi_example.py   # OpenAI CLIP ViT-B/32 + Laplace noise
```

Both scripts hook into the model, add noise, build simple sensitive masks, and call `assess_privacy_budget_from_features`. Replace the dummy inputs and sensitivity logic with your own data and privacy definitions.

---

## Generating `detector_metrics.csv` and `vlm_metrics.csv`

To produce the per-run CSVs used for paper tables (e.g. Table~\\ref{tab:rmse}):

1. **Install script dependencies** (models are downloaded from Hugging Face / ultralytics; no local weights):

   ```bash
   pip install -r requirements.txt
   ```

   This adds `torch`, `ultralytics`, `opencv-python`, `transformers`, `timm`, etc.

2. **Run the detector and VLM scripts** from `src`:

   ```bash
   cd src
   python scripts/run_detector_metrics.py   # YOLO for MDCRF/PPDPTS, DETR from HF
   python scripts/run_vlm_metrics.py       # CLIP and BLIP from HF
   ```

   Outputs: `results/detector_metrics.csv`, `results/vlm_metrics.csv` (columns: model, epsilon, seed, chi2, kl, mmd, rmse, wass1, empa_bias_bua, empa_bias_tda).

3. **Aggregate to LaTeX** (optional):

   ```bash
   python scripts/aggregate_metrics.py --csv ../results/detector_metrics.csv --mode detector
   python scripts/aggregate_metrics.py --csv ../results/vlm_metrics.csv --mode vlm
   ```

   Redirect output to `results/detector_tab_rmse.tex` and `results/vlm_tab_results.tex` if you use them in the paper.

---

## License and contact

See the repository and submodule repositories for license and contact information.
