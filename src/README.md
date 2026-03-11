# Bodhi VLM experiment scripts

This directory contains the Python scripts used for Bodhi VLM experiments and integration examples. They implement:

- BUA/TDA-style grouping of hierarchical features,
- EMPA-style privacy budget assessment,
- Synthetic experiments comparing EMPA with Chi-square, K-L divergence, and MMD,
- Example integrations with YOLO and CLIP.

## Dependencies

From the project root, install the core dependencies:

```bash
pip install -r requirements.txt
```

This installs the packages required for synthetic experiments and plotting (`numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`).

To run the integration examples you will also need:

```bash
pip install ultralytics
pip install git+https://github.com/openai/CLIP.git
```

## Running synthetic experiments

The main synthetic experiment runner is `run_experiments.py`:

```bash
cd src

# Default: epsilon = 0.1, 0.01; results written to ../results
python run_experiments.py

# Custom output directory and multiple budgets
python run_experiments.py --out_dir ../results --epsilon 0.1 0.01 0.001

# More samples and layers (closer to VLM-style hierarchies)
python run_experiments.py --n_samples 500 --n_layers 6 --epsilon 0.1 0.01
```

Outputs include:

- `bodhi_vlm_metrics.csv` (per-ε metrics: chi2, kl, mmd, rmse, empa_bias_bua, empa_bias_tda)
- `bodhi_vlm_empa_bias.png` and `bodhi_vlm_metrics_vs_epsilon.png` (if `matplotlib` is available)

## Core modules

- `metrics.py`: Chi-square, K-L divergence, MMD, rMSE, and EMPA-style bias (simplified EM mixture weights).
- `grouping.py`: BUA/TDA-style grouping (MDAV-like clustering with sensitive/non-sensitive partitions).
- `run_experiments.py`: synthetic multi-layer features, privacy noise injection, grouping and metric comparison.
- `bodhi_vlm_pipeline.py`: high-level API that combines BUA/TDA and EMPA into `assess_privacy_budget_from_features`, suitable for arbitrary backbone features.

## Integration examples (YOLO / CLIP)

The following scripts demonstrate how to plug Bodhi VLM into real models:

- `yolo_bodhi_example.py`: hooks into ultralytics YOLOv8 backbone layers, constructs a simple sensitive mask, and calls `assess_privacy_budget_from_features`.
- `clip_bodhi_example.py`: hooks into OpenAI CLIP ViT-B/32 transformer blocks to extract patch features, marks some images as sensitive, and calls `assess_privacy_budget_from_features`.

To run the examples (after installing the optional dependencies):

```bash
cd src
python yolo_bodhi_example.py
python clip_bodhi_example.py
```

