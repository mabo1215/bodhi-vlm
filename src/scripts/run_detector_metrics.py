# -*- coding: utf-8 -*-
"""
Compatibility entrypoint: equivalent to main.py + config.experiments.detector.
Preferred usage is to enable `detector` in config.json and run `python main.py`.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.detector import run as run_detector


def main():
    parser = argparse.ArgumentParser(description="Generate detector_metrics.csv")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.1, 0.01])
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = args.out_dir or os.path.join(root, "results")
    config = {"seeds": args.seeds, "epsilons": args.epsilons, "num_images": args.num_images, "size": args.size}
    run_detector(config, out_dir)


if __name__ == "__main__":
    main()
