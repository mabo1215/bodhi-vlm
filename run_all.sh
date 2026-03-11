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
# Download test images for detector (YOLO/DETR) so runs yield real detections
if [ -f "src/scripts/download_detector_test_images.py" ]; then
  "$PYTHON" src/scripts/download_detector_test_images.py 6 || true
fi
"$PYTHON" src/main.py --config src/config.json

# Copy synthetic and decomposition figures into paper for inclusion
if [ -d "results" ] && [ -d "paper/images" ]; then
  for f in bodhi_vlm_empa_bias.png bodhi_vlm_metrics_vs_epsilon.png decomposition_fixed_partition_bias_vs_epsilon.png; do
    [ -f "results/$f" ] && cp "results/$f" "paper/images/$f" && echo "Copied results/$f -> paper/images/"
  done
fi

echo "Done. Outputs are in results/. Figures used by the paper are in paper/images/."
