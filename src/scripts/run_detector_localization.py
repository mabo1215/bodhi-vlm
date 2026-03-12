# -*- coding: utf-8 -*-
"""
N4: Localization evaluation for detector backbones.
Computes IoU between (1) predicted sensitive mask from BUA (thresholded sensitivity score)
and (2) ground-truth mask from detection boxes (cells overlapping any box).

Usage (from repo root):
  PYTHONPATH=src python src/scripts/run_detector_localization.py [--num_images 5] [--out_dir results]

Requires: test images in data/test_images/, YOLO (ultralytics). Outputs: localization_iou.csv
"""
import argparse
import os
import sys
import csv

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

from experiments.detector import _load_test_images, _add_noise_to_images
from utils.grouping import bua_style
from core.pipeline import assess_privacy_budget_from_features


def _boxes_to_mask_float(boxes_xyxy: np.ndarray, H: int, W: int, im_h: int, im_w: int) -> np.ndarray:
    """Rasterize boxes (x1,y1,x2,y2 in image coords) to grid (H,W). Value 1 where cell center inside any box."""
    mask = np.zeros((H, W), dtype=float)
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return mask
    for x1, y1, x2, y2 in boxes_xyxy:
        # Map to grid [0, H-1], [0, W-1]
        gx1 = int(np.clip(x1 / im_w * W, 0, W - 1))
        gx2 = int(np.clip(x2 / im_w * W, 0, W - 1))
        gy1 = int(np.clip(y1 / im_h * H, 0, H - 1))
        gy2 = int(np.clip(y2 / im_h * H, 0, H - 1))
        mask[gy1:gy2 + 1, gx1:gx2 + 1] = 1.0
    return mask


def _iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """IoU = intersection / union (binary masks)."""
    pred = (np.asarray(mask_pred).flatten() > 0.5).astype(float)
    gt = (np.asarray(mask_gt).flatten() > 0.5).astype(float)
    inter = (pred * gt).sum()
    union = (pred + gt > 0).sum()
    if union < 1e-10:
        return float("nan")
    return float(inter / union)


def run_localization(num_images: int = 5, size: int = 640, test_images_dir: str = "data/test_images", out_dir: str = "results"):
    """Run localization eval: get boxes from YOLO, feature map with shape, BUA sensitive = 90th percentile score proxy, IoU."""
    try:
        from ultralytics import YOLO
        from models import ensure_yolo_weights
    except ImportError:
        print("run_detector_localization: ultralytics or models not available; skip.")
        return
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abs_dir = test_images_dir if os.path.isabs(test_images_dir) else os.path.join(_PROJECT_ROOT, test_images_dir)
    images = _load_test_images(abs_dir, num_images, size, device)
    if images is None:
        print("No test images; using random. Localization IoU will be random.")
        images = torch.rand(num_images, 3, size, size, device=device)
    weights_path = ensure_yolo_weights("yolov8n.pt")
    model = YOLO(weights_path)
    model.to(device)
    model.eval()

    # Get detections (boxes) and feature map with shape
    feats_list = []
    shapes_list = []

    def hook_fn(m, inp, out):
        x = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            b, c, h, w = x.shape
            feats_list.append(x.permute(0, 2, 3, 1).reshape(b * h * w, c).detach().cpu().numpy())
            shapes_list.append((int(h), int(w)))

    handle = None
    for name, module in model.model.named_modules():
        if "backbone" in name and any(k in name for k in ["cv2", "cv3", "cv4"]):
            handle = module.register_forward_hook(hook_fn)
            break
    with torch.no_grad():
        results = model(images)
    if handle:
        handle.remove()
    if not feats_list or not shapes_list:
        print("Could not get feature map with shape; skipping localization.")
        return

    feats = feats_list[0]
    H, W = shapes_list[0]
    n_cells = H * W

    # GT mask from boxes (person class only for MOT-style)
    boxes_all = []
    for r in results:
        if r.boxes is None or r.boxes.xyxy is None:
            boxes_all.append(np.zeros((0, 4)))
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        boxes_all.append(xyxy)
    # Single image for simplicity
    im_h, im_w = size, size
    mask_gt = np.zeros((H, W))
    if len(boxes_all) > 0 and len(boxes_all[0]) > 0:
        mask_gt = _boxes_to_mask_float(boxes_all[0], H, W, im_h, im_w)

    # Predicted sensitive mask: use 90th percentile of L2 norm of features as proxy for "sensitivity"
    scores = np.linalg.norm(feats, axis=1)
    thresh = np.percentile(scores, 90)
    sens_idx = np.where(scores >= thresh)[0]
    mask_pred = np.zeros((H, W))
    for idx in sens_idx:
        if idx < n_cells:
            h, w = divmod(idx, W)
            mask_pred[h, w] = 1.0

    iou = _iou(mask_pred, mask_gt)
    row = {"iou": iou, "H": H, "W": W, "n_sensitive_cells": int(sens_idx.size), "n_gt_cells": int(mask_gt.sum())}
    path = os.path.join(out_dir, "localization_iou.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"Localization IoU (proxy): {iou:.4f} -> {path}")


def main():
    p = argparse.ArgumentParser(description="N4: Detector localization IoU (BUA sensitive vs box mask)")
    p.add_argument("--num_images", type=int, default=5)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--test_images_dir", type=str, default="data/test_images")
    p.add_argument("--out_dir", type=str, default="results")
    args = p.parse_args()
    run_localization(
        num_images=args.num_images,
        size=args.size,
        test_images_dir=args.test_images_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
