# -*- coding: utf-8 -*-
"""
Compatibility entrypoint: equivalent to main.py + config.experiments.aggregate.
Preferred usage is to enable `aggregate` in config.json and run `python main.py`.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to per-run metrics CSV")
    parser.add_argument("--mode", type=str, choices=["detector", "vlm"], required=True)
    parser.add_argument("--out_dir", type=str, default=None, help="Where to write .tex (default: dir of CSV)")
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    # 根据 mode 决定用哪个 CSV；aggregate.run 会读 detector_csv 和/或 vlm_csv
    config = {}
    if args.mode == "detector":
        config["detector_csv"] = args.csv
        config["vlm_csv"] = ""
    else:
        config["detector_csv"] = ""
        config["vlm_csv"] = args.csv
    from experiments.aggregate import run as run_aggregate
    run_aggregate(config, out_dir)


if __name__ == "__main__":
    main()
