# -*- coding: utf-8 -*-
"""
Run only the component-ablation block (full / bua_only / tda_only / no_empa) with 5 seeds.
Writes results/ablation_component.csv and results/ablation_component_summary.csv.
Use when detector main run already completed and you only need to (re)run ablation.
"""
import os
import sys
import json
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from experiments.detector import run_component_ablation, run

def main():
    parser = argparse.ArgumentParser(description="Run detector component ablation only (5 seeds x 4 configs x 2 eps)")
    parser.add_argument("--config", type=str, default=os.path.join(_ROOT, "src", "config.json"))
    parser.add_argument("--out_dir", type=str, default=os.path.join(_ROOT, "results"))
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    detector_config = config.get("experiments", {}).get("detector", {})
    detector_config.setdefault("seeds", [0, 1, 2, 3, 4])
    detector_config.setdefault("epsilons", [0.1, 0.01])
    detector_config.setdefault("ablation_model", "DETR")
    os.makedirs(args.out_dir, exist_ok=True)
    import torch
    from experiments.detector import _load_test_images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_images = detector_config.get("num_images", 10)
    size = detector_config.get("size", 640)
    test_images_dir = detector_config.get("test_images_dir", "data/test_images")
    abs_dir = test_images_dir if os.path.isabs(test_images_dir) else os.path.join(_ROOT, test_images_dir)
    images = _load_test_images(abs_dir, num_images, size, device)
    if images is None:
        print("No test images; using random tensors.")
        torch.manual_seed(42)
        images = torch.rand(num_images, 3, size, size, device=device)
    run_component_ablation(detector_config, args.out_dir, device, images)
    print("Done.")

if __name__ == "__main__":
    main()
