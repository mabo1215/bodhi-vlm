# -*- coding: utf-8 -*-
"""
Compatibility entrypoint: equivalent to main.py + config.experiments.interpretability.
Preferred usage is to enable `interpretability` in config.json and run `python main.py`.
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    parser = argparse.ArgumentParser(description="Bodhi VLM interpretability figures")
    parser.add_argument("--out_dir", type=str, default="paper/images")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    config = {"n_samples": args.n_samples, "n_layers": args.n_layers, "epsilon": args.epsilon}
    from experiments.interpretability import run as run_interpretability
    run_interpretability(config, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
