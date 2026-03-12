#!/usr/bin/env bash
# Run all experiments and generate all data (CSV + figures).
# All models (*.pt, Hugging Face, torch.hub) are downloaded under src/models/.
# Uses src/config.json (all experiments enabled). From repo root: ./run_all.sh or bash run_all.sh

set -e
cd "$(dirname "$0")"

if [ -f ".venv/Scripts/python.exe" ]; then
  PYTHON=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python"
fi

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
# Use existing images in data/test_images/ (no auto-download). To fetch images manually: python src/scripts/download_detector_test_images.py 6
"$PYTHON" src/main.py --config src/config.json

# Copy figures used by the paper (synthetic + decomposition + interpretability)
if [ -d "results" ] && [ -d "paper/images" ]; then
  for f in bodhi_vlm_empa_bias.png bodhi_vlm_metrics_vs_epsilon.png decomposition_fixed_partition_bias_vs_epsilon.png bodhi_vlm_sensitive_dist.png bodhi_vlm_tsne.png; do
    [ -f "results/$f" ] && cp "results/$f" "paper/images/$f" && echo "Copied results/$f -> paper/images/"
  done
fi

echo "Done. Outputs are in results/. Figures used by the paper are in paper/images/."
