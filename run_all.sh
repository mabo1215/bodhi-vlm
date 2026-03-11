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
if [ -f "scripts/download_detector_test_images.py" ]; then
  "$PYTHON" scripts/download_detector_test_images.py 6 || true
fi
"$PYTHON" src/main.py --config src/config.json

echo "Done. Outputs are in results/."
