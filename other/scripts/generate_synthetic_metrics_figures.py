#!/usr/bin/env python3
"""
Generate synthetic Bodhi VLM figures used in the paper:

  - bodhi_vlm_empa_bias.png
  - bodhi_vlm_metrics_vs_epsilon.png
  - bodhi_vlm_sensitive_dist.png
  - bodhi_vlm_tsne.png

This script is a thin wrapper around the core experiment
and interpretability scripts in `src/`. It is provided
for convenience so that all paper figures can be
re-generated from the `paper/scripts` directory.

Usage (from project root or paper/):

  python paper/scripts/generate_synthetic_metrics_figures.py
"""

import os
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def main() -> None:
    root = project_root()
    env = os.environ.copy()
    python_exe = sys.executable or "python"

    # 1) Run synthetic experiments: bodhi_vlm_metrics.csv +
    #    bodhi_vlm_empa_bias.png + bodhi_vlm_metrics_vs_epsilon.png
    cmd_exp = [
        python_exe,
        str(root / "src" / "run_experiments.py"),
        "--out_dir",
        str(root / "paper" / "images"),
    ]
    print("Running:", " ".join(cmd_exp))
    subprocess.run(cmd_exp, cwd=str(root), check=True, env=env)

    # 2) Run interpretability plots: bodhi_vlm_sensitive_dist.png,
    #    bodhi_vlm_tsne.png
    cmd_interp = [
        python_exe,
        str(root / "paper" / "scripts" / "plot_interpretability.py"),
        "--out_dir",
        str(root / "paper" / "images"),
    ]
    print("Running:", " ".join(cmd_interp))
    subprocess.run(cmd_interp, cwd=str(root), check=True, env=env)

    print("All synthetic Bodhi VLM figures generated in paper/images/")


if __name__ == "__main__":
    main()

