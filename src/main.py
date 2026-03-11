# -*- coding: utf-8 -*-
"""
Unified entrypoint for Bodhi VLM.

Experiments are controlled by config.json and executed via:

  python main.py
  python main.py --config config.json
  python main.py --config config.json --out_dir ../results
  python main.py --experiments synthetic aggregate

When running from the project root, either `cd src` first or use `python src/main.py`.
"""
import argparse
import json
import os
import sys

# Ensure we can import utils, core, experiments when running from src
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def load_config(config_path: str) -> dict:
    path = os.path.abspath(config_path)
    if not os.path.exists(path):
        path = os.path.join(_SCRIPT_DIR, config_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Bodhi VLM: config-driven experiments")
    parser.add_argument("--config", type=str, default=os.path.join(_SCRIPT_DIR, "config.json"), help="Path to config file")
    parser.add_argument("--out_dir", type=str, default=None, help="Override out_dir in config")
    parser.add_argument("--experiments", type=str, nargs="+", default=None, help="Only run selected experiments, e.g. synthetic detector vlm aggregate interpretability")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = args.out_dir or config.get("out_dir", "results")
    # If out_dir is relative, interpret it as relative to the project root,
    # not the current working directory, so that all outputs live under
    # the repository-level `results/` by default.
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_PROJECT_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    experiments_config = config.get("experiments", {})
    run_only = args.experiments  # if provided, run only these names
    runners = {
        "synthetic": ("experiments.synthetic", "run"),
        "detector": ("experiments.detector", "run"),
        "vlm": ("experiments.vlm", "run"),
        "aggregate": ("experiments.aggregate", "run"),
        "interpretability": ("experiments.interpretability", "run"),
    }

    for name, (module_path, func_name) in runners.items():
        if run_only and name not in run_only:
            continue
        exp = experiments_config.get(name, {})
        if not exp.get("enabled", False):
            continue
        print(f"\n[{name}]")
        try:
            mod = __import__(module_path, fromlist=[func_name])
            run_fn = getattr(mod, func_name)
            run_fn(exp, out_dir)
        except Exception as e:
            print(f"[{name}] Error: {e}")
            raise

    print("\nDone.")


if __name__ == "__main__":
    main()
