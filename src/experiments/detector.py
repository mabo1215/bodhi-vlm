# -*- coding: utf-8 -*-
"""Detector experiments: YOLO (MDCRF/PPDPTS) + DETR, write detector_metrics.csv."""
import os
import csv
import glob
import numpy as np
import torch
from core.pipeline import assess_privacy_budget_from_features
from models import HF_CACHE, ensure_yolo_weights

MAX_POINTS_PER_LAYER = 2500

# Project root (parent of src)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))


def _subsample_layers(feats_list: list, max_points: int = MAX_POINTS_PER_LAYER, seed: int = 0) -> list:
    out = []
    rng = np.random.default_rng(seed)
    for f in feats_list:
        f = np.asarray(f)
        n = f.shape[0]
        out.append(f if n <= max_points else f[rng.choice(n, size=max_points, replace=False)])
    return out


def _add_noise_to_images(images: torch.Tensor, epsilon: float, seed: int, scale: float = 1.0) -> torch.Tensor:
    sigma = scale / (epsilon + 1e-8)
    g = torch.Generator(device=images.device).manual_seed(seed)
    return torch.clamp(images + torch.randn_like(images, generator=g) * sigma, 0.0, 1.0)


def _collect_yolo_features(model, images: torch.Tensor) -> list:
    feats_per_layer, handles = [], []

    def make_hook():
        def hook(m, inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(x, torch.Tensor) or x.dim() != 4:
                return
            b, c, h, w = x.shape
            feats_per_layer.append(x.permute(0, 2, 3, 1).reshape(b * h * w, c).detach().cpu().numpy())
        return hook

    # Pipeline expects all layers to have the same feature dim; use a single backbone layer for YOLO.
    # Primary: legacy naming (backbone + cv2/cv3/cv4) – hook only one layer if multiple match
    for name, module in model.model.named_modules():
        if "backbone" in name and any(k in name for k in ["cv2", "cv3", "cv4"]):
            handles.append(module.register_forward_hook(make_hook()))
            break
    # Fallback: YOLOv8 backbone is model.model.model (Sequential); use index 6 only (same dim for concat)
    backbone = getattr(model.model, "model", None)
    if not handles and backbone is not None and hasattr(backbone, "__len__") and len(backbone) >= 7:
        handles.append(backbone[6].register_forward_hook(make_hook()))
    if not handles:
        for name, module in model.model.named_modules():
            if "backbone" in name.lower() and len(list(module.children())) > 0:
                handles.append(module.register_forward_hook(make_hook()))
                break
    with torch.no_grad():
        _ = model(images)
    for h in handles:
        h.remove()
    return feats_per_layer


def _collect_detr_features(model, processor, images: torch.Tensor, device: str) -> list:
    feats_list = []
    # Images are already in [0, 1]; avoid double rescale warning
    imgs_np = [images[i].cpu().permute(1, 2, 0).numpy() for i in range(images.shape[0])]
    inp = processor(images=imgs_np, return_tensors="pt", do_rescale=False)
    pixel_values = inp["pixel_values"].to(device)

    def hook_fn(m, inp, out):
        x = out[0] if isinstance(out, (tuple, list)) else (out if isinstance(out, torch.Tensor) else list(out.values())[-1])
        if not isinstance(x, torch.Tensor):
            x = x[0] if isinstance(x, (tuple, list)) else x
        b, c, h, w = x.shape
        feats_list.append(x.permute(0, 2, 3, 1).reshape(b * h * w, c).detach().cpu().numpy())

    handle = model.model.backbone.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(pixel_values=pixel_values)
    handle.remove()
    return feats_list


def _load_test_images(test_images_dir: str, num_images: int, size: int, device: torch.device):
    """Load and resize images from test_images_dir. Returns (N, 3, size, size) in [0,1] or None."""
    abs_dir = test_images_dir if os.path.isabs(test_images_dir) else os.path.join(_PROJECT_ROOT, test_images_dir)
    if not os.path.isdir(abs_dir):
        return None
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(abs_dir, ext)))
    paths = sorted(paths)[:num_images]
    if len(paths) < num_images:
        return None
    try:
        from torchvision.io import read_image
        import torch.nn.functional as F
    except ImportError:
        return None
    tensors = []
    for p in paths:
        x = read_image(p)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] == 4:
            x = x[:3]
        x = x.float().unsqueeze(0) / 255.0
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        tensors.append(x.squeeze(0))
    return torch.stack(tensors, dim=0).to(device)


def _sensitive_masks(layer_features: list, has_sensitive: bool) -> list:
    return [np.ones(f.shape[0], dtype=bool) if has_sensitive else np.zeros(f.shape[0], dtype=bool) for f in layer_features]


def _run_yolo(model_name: str, device: str, images: torch.Tensor, epsilon: float, seed: int):
    from ultralytics import YOLO
    weights_path = ensure_yolo_weights("yolov8n.pt")
    model = YOLO(weights_path)
    model.to(device)
    model.eval()
    raw_orig = _collect_yolo_features(model, images)
    if not raw_orig:
        raise ValueError("no YOLO backbone layers found for feature extraction")
    raw_noised = _collect_yolo_features(model, _add_noise_to_images(images, epsilon, seed))
    if not raw_noised:
        raise ValueError("no YOLO backbone layers found for noised features")
    feats_orig = _subsample_layers(raw_orig, seed=seed)
    feats_noised = _subsample_layers(raw_noised, seed=seed)
    masks = _sensitive_masks(feats_noised, True)
    return assess_privacy_budget_from_features(feats_orig, feats_noised, masks, epsilon=epsilon, bins=20, k_mdav=3)


def _run_detr(device: str, images: torch.Tensor, epsilon: float, seed: int):
    import logging
    from transformers import DetrForObjectDetection, DetrImageProcessor
    cache = HF_CACHE
    # Suppress expected "UNEXPECTED" load report (num_batches_tracked from BN layers)
    log = logging.getLogger("transformers.modeling_utils")
    old_level = log.level
    log.setLevel(logging.WARNING)
    try:
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", cache_dir=cache)
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", cache_dir=cache)
    finally:
        log.setLevel(old_level)
    model.to(device)
    model.eval()
    feats_orig = _subsample_layers(_collect_detr_features(model, processor, images, device), seed=seed)
    images_noised = _add_noise_to_images(images, epsilon, seed)
    feats_noised = _subsample_layers(_collect_detr_features(model, processor, images_noised, device), seed=seed)
    masks = _sensitive_masks(feats_noised, True)
    return assess_privacy_budget_from_features(feats_orig, feats_noised, masks, epsilon=epsilon, bins=20, k_mdav=3)


def run(config: dict, out_dir: str) -> None:
    """
    Run detector experiments.

    Config keys:
      - seeds: list of ints
      - epsilons: list of floats
      - num_images: int
      - size: int (image side length)
      - test_images_dir: str (e.g. "data/test_images") – if set and dir exists, use real images for detections

    Writes detector_metrics.csv into out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    seeds = config.get("seeds", [0, 1, 2, 3, 4])
    epsilons = config.get("epsilons", [0.1, 0.01])
    num_images = config.get("num_images", 4)
    size = config.get("size", 640)
    test_images_dir = config.get("test_images_dir", "data/test_images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    images = _load_test_images(test_images_dir, num_images, size, device)
    if images is None:
        print("  No test images found (run scripts/download_detector_test_images.py); using random tensors.")
        images = torch.rand(num_images, 3, size, size, device=device)
    else:
        print(f"  Using {num_images} test images from {test_images_dir}")

    fieldnames = ["model", "epsilon", "seed", "chi2", "kl", "mmd", "rmse", "wass1", "empa_bias_bua", "empa_bias_tda"]
    rows = []
    nan_row = lambda model, eps, s: {"model": model, "epsilon": eps, "seed": s, "chi2": np.nan, "kl": np.nan, "mmd": np.nan, "rmse": np.nan, "wass1": np.nan, "empa_bias_bua": np.nan, "empa_bias_tda": np.nan}

    for epsilon in epsilons:
        for seed in seeds:
            for model_name in ["MDCRF", "PPDPTS"]:
                try:
                    m = _run_yolo(model_name, device, images, epsilon, seed)
                    rows.append({"model": model_name, "epsilon": epsilon, "seed": seed, "chi2": m["chi2"], "kl": m["kl"], "mmd": m["mmd"], "rmse": m["rmse"], "wass1": m.get("wass1", np.nan), "empa_bias_bua": m["empa_bias_bua"], "empa_bias_tda": m["empa_bias_tda"]})
                    print(f"  Done {model_name} eps={epsilon} seed={seed}")
                except Exception as e:
                    print(f"  Error {model_name} eps={epsilon} seed={seed}: {e}")
                    rows.append(nan_row(model_name, epsilon, seed))
            try:
                m = _run_detr(device, images, epsilon, seed)
                rows.append({"model": "DETR", "epsilon": epsilon, "seed": seed, "chi2": m["chi2"], "kl": m["kl"], "mmd": m["mmd"], "rmse": m["rmse"], "wass1": m.get("wass1", np.nan), "empa_bias_bua": m["empa_bias_bua"], "empa_bias_tda": m["empa_bias_tda"]})
                print(f"  Done DETR eps={epsilon} seed={seed}")
            except Exception as e:
                print(f"  Error DETR eps={epsilon} seed={seed}: {e}")
                rows.append(nan_row("DETR", epsilon, seed))

    csv_path = os.path.join(out_dir, "detector_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")
