# -*- coding: utf-8 -*-
"""Detector experiments: YOLO (MDCRF/PPDPTS) + DETR, write detector_metrics.csv."""
import os
import csv
import glob
import warnings
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
    # randn_like() does not accept generator; use randn(..., generator=g) with same shape
    noise = torch.randn(images.shape, device=images.device, dtype=images.dtype, generator=g)
    return torch.clamp(images + noise * sigma, 0.0, 1.0)


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


def _first_4d_tensor(obj):
    """Extract the first 4D (B,C,H,W) tensor from nested tuple/list/dict from DETR backbone output."""
    if isinstance(obj, torch.Tensor) and obj.dim() == 4:
        return obj
    if isinstance(obj, torch.Tensor):
        return None
    if isinstance(obj, (tuple, list)):
        for item in obj:
            t = _first_4d_tensor(item)
            if t is not None:
                return t
        return None
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            t = _first_4d_tensor(obj[k])
            if t is not None:
                return t
        return None
    return None


def _collect_detr_features(model, processor, images: torch.Tensor, device: str) -> list:
    feats_list = []
    # Images are already in [0, 1]; avoid double rescale warning
    imgs_np = [images[i].cpu().permute(1, 2, 0).numpy() for i in range(images.shape[0])]
    inp = processor(images=imgs_np, return_tensors="pt", do_rescale=False)
    pixel_values = inp["pixel_values"].to(device)

    def hook_fn(m, inp, out):
        x = _first_4d_tensor(out)
        if x is None:
            return
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
    # Prefer test_01.jpg ... test_NN.jpg if all present (from copy_test_images_as_named.py)
    preferred = [os.path.join(abs_dir, f"test_{i:02d}.jpg") for i in range(1, num_images + 1)]
    if all(os.path.isfile(p) for p in preferred):
        paths = preferred
    else:
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


def _run_yolo(model_name: str, device: str, images: torch.Tensor, epsilon: float, seed: int, ablation_mode: str = "full"):
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
    return assess_privacy_budget_from_features(
        feats_orig, feats_noised, masks, epsilon=epsilon, bins=20, k_mdav=3, ablation_mode=ablation_mode
    )


def _run_detr(device: str, images: torch.Tensor, epsilon: float, seed: int, ablation_mode: str = "full"):
    import logging
    from transformers import DetrForObjectDetection, DetrImageProcessor
    cache = HF_CACHE
    # UNEXPECTED: checkpoint has num_batches_tracked (BatchNorm buffers) from ResNet backbone;
    # HF DetrForObjectDetection does not register them, so they are skipped. Safe to ignore.
    # Suppress LOAD REPORT by raising transformers log level during load.
    tlog = logging.getLogger("transformers")
    old_level = tlog.level
    tlog.setLevel(logging.ERROR)
    # Suppress PyTorch "copying from non-meta to meta parameter" when loading ResNet backbone
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*copying from a non-meta parameter.*meta parameter.*",
            category=UserWarning,
            module="torch.nn.modules.module",
        )
        try:
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", cache_dir=cache)
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", cache_dir=cache)
        finally:
            tlog.setLevel(old_level)
    model.to(device)
    model.eval()
    feats_orig = _subsample_layers(_collect_detr_features(model, processor, images, device), seed=seed)
    images_noised = _add_noise_to_images(images, epsilon, seed)
    feats_noised = _subsample_layers(_collect_detr_features(model, processor, images_noised, device), seed=seed)
    masks = _sensitive_masks(feats_noised, True)
    return assess_privacy_budget_from_features(
        feats_orig, feats_noised, masks, epsilon=epsilon, bins=20, k_mdav=3, ablation_mode=ablation_mode
    )


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

    if config.get("run_component_ablation", False):
        run_component_ablation(config, out_dir, device, images)


def run_component_ablation(config: dict, out_dir: str, device: torch.device, images: torch.Tensor) -> None:
    """Run component ablation (full, bua_only, tda_only, no_empa) with 5 seeds. Use DETR (no cv2); PPDPTS if config has ablation_model='PPDPTS'. Write ablation_component.csv and ablation_component_summary.csv."""
    seeds = config.get("seeds", [0, 1, 2, 3, 4])
    epsilons = config.get("epsilons", [0.1, 0.01])
    modes = ["full", "bua_only", "tda_only", "no_empa"]
    ablation_model = config.get("ablation_model", "DETR")  # DETR works without cv2
    rows = []
    for epsilon in epsilons:
        for seed in seeds:
            for mode in modes:
                try:
                    if ablation_model == "DETR":
                        m = _run_detr(device, images, epsilon, seed, ablation_mode=mode)
                    else:
                        m = _run_yolo(ablation_model, device, images, epsilon, seed, ablation_mode=mode)
                    rows.append({
                        "epsilon": epsilon, "seed": seed, "config": mode,
                        "deviation": m["deviation"], "rmse_budget": m["rmse_budget"],
                    })
                except Exception as e:
                    print(f"  Ablation error {mode} eps={epsilon} seed={seed}: {e}")
                    rows.append({"epsilon": epsilon, "seed": seed, "config": mode, "deviation": np.nan, "rmse_budget": np.nan})
    csv_path = os.path.join(out_dir, "ablation_component.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epsilon", "seed", "config", "deviation", "rmse_budget"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    try:
        from utils.metrics import confidence_interval_95
    except ImportError:
        confidence_interval_95 = None
    summary_rows = []
    for config_name in modes:
        for eps in epsilons:
            vals_dev = [r["deviation"] for r in rows if r["config"] == config_name and r["epsilon"] == eps and not np.isnan(r["deviation"])]
            vals_rmse = [r["rmse_budget"] for r in rows if r["config"] == config_name and r["epsilon"] == eps and not np.isnan(r["rmse_budget"])]
            n = len(vals_dev)
            mean_dev = np.mean(vals_dev) if vals_dev else np.nan
            std_dev = np.std(vals_dev, ddof=1) if n > 1 and vals_dev else 0.0
            mean_rmse = np.mean(vals_rmse) if vals_rmse else np.nan
            std_rmse = np.std(vals_rmse, ddof=1) if n > 1 and vals_rmse else 0.0
            ci_dev = confidence_interval_95(np.array(vals_dev)) if confidence_interval_95 and len(vals_dev) >= 2 else (mean_dev, mean_dev)
            ci_rmse = confidence_interval_95(np.array(vals_rmse)) if confidence_interval_95 and len(vals_rmse) >= 2 else (mean_rmse, mean_rmse)
            summary_rows.append({
                "config": config_name, "epsilon": eps,
                "dev_mean": mean_dev, "dev_std": std_dev, "dev_ci_low": ci_dev[0], "dev_ci_high": ci_dev[1],
                "rmse_mean": mean_rmse, "rmse_std": std_rmse, "rmse_ci_low": ci_rmse[0], "rmse_ci_high": ci_rmse[1],
            })
    summary_path = os.path.join(out_dir, "ablation_component_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config", "epsilon", "dev_mean", "dev_std", "dev_ci_low", "dev_ci_high", "rmse_mean", "rmse_std", "rmse_ci_low", "rmse_ci_high"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {summary_path}")
