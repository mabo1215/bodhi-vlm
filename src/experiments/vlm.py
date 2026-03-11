# -*- coding: utf-8 -*-
"""VLM experiments: CLIP/BLIP (Hugging Face) producing vlm_metrics.csv."""
import os
import csv
import numpy as np
import torch
from core.pipeline import assess_privacy_budget_from_features

MAX_POINTS_PER_LAYER = 2500


def _subsample_layers(feats_list: list, max_points: int = MAX_POINTS_PER_LAYER, seed: int = 0) -> list:
    out = []
    rng = np.random.default_rng(seed)
    for f in feats_list:
        f = np.asarray(f)
        n = f.shape[0]
        out.append(f if n <= max_points else f[rng.choice(n, size=max_points, replace=False)])
    return out


def _add_noise_to_images(images: torch.Tensor, epsilon: float, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator(device=images.device).manual_seed(seed)
    sigma = scale / (epsilon + 1e-8)
    return torch.clamp(images + torch.randn_like(images, generator=g) * sigma, 0.0, 1.0)


def _sensitive_masks(layer_features: list, has_sensitive: bool) -> list:
    return [np.ones(f.shape[0], dtype=bool) if has_sensitive else np.zeros(f.shape[0], dtype=bool) for f in layer_features]


def _collect_clip_layers(model, pixel_values: torch.Tensor, device: str) -> list:
    feats_per_layer = []
    vision = model.vision_model
    if not hasattr(vision, "encoder") or not hasattr(vision.encoder, "layers"):
        with torch.no_grad():
            out = vision(pixel_values)
        h = out.last_hidden_state
        x = h[:, 1:, :] if h.shape[1] > 1 else h
        b, p, d = x.shape
        feats_per_layer.append(x.reshape(b * p, d).cpu().numpy())
        return feats_per_layer
    def make_hook():
        def hook(m, inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            if x.shape[1] > 1:
                x = x[:, 1:, :]
            b, p, d = x.shape
            feats_per_layer.append(x.reshape(b * p, d).detach().cpu().numpy())
        return hook
    handles = [layer.register_forward_hook(make_hook()) for layer in vision.encoder.layers]
    with torch.no_grad():
        _ = vision(pixel_values)
    for h in handles:
        h.remove()
    return feats_per_layer


def _collect_blip_layers(model, pixel_values: torch.Tensor, device: str) -> list:
    feats_per_layer = []
    vision = model.vision_model
    layers = getattr(getattr(vision, "encoder", None), "layer", None) or getattr(getattr(vision, "encoder", None), "layers", None)
    if layers is None:
        with torch.no_grad():
            out = vision(pixel_values)
        h = getattr(out, "last_hidden_state", out[0] if isinstance(out, (tuple, list)) else out)
        if isinstance(h, (tuple, list)):
            h = h[0]
        if h.dim() == 3:
            b, s, d = h.shape
            feats_per_layer.append(h.reshape(b * s, d).cpu().numpy())
        return feats_per_layer
    def make_hook():
        def hook(m, inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            if x.dim() == 3:
                b, s, d = x.shape
                feats_per_layer.append(x.reshape(b * s, d).detach().cpu().numpy())
        return hook
    handles = [layer.register_forward_hook(make_hook()) for layer in layers]
    with torch.no_grad():
        _ = vision(pixel_values)
    for h in handles:
        h.remove()
    return feats_per_layer


def _run_clip(device: str, images: torch.Tensor, epsilon: float, seed: int, processor):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = processor or CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()
    imgs_np = [images[i].cpu().permute(1, 2, 0).numpy() for i in range(images.shape[0])]
    inp = proc(images=imgs_np, return_tensors="pt")
    pixel_values = inp["pixel_values"].to(device)
    feats_orig = _subsample_layers(_collect_clip_layers(model, pixel_values, device), seed=seed)
    images_noised = _add_noise_to_images(images, epsilon, seed)
    inp_n = proc(images=[images_noised[i].cpu().permute(1, 2, 0).numpy() for i in range(images_noised.shape[0])], return_tensors="pt")
    feats_noised = _subsample_layers(_collect_clip_layers(model, inp_n["pixel_values"].to(device), device), seed=seed)
    masks = _sensitive_masks(feats_noised, True)
    return assess_privacy_budget_from_features(feats_orig, feats_noised, masks, epsilon=epsilon, bins=20, k_mdav=3)


def _run_blip(device: str, images: torch.Tensor, epsilon: float, seed: int, processor):
    from transformers import BlipForImageTextRetrieval, BlipProcessor
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-base-en")
    proc = processor or BlipProcessor.from_pretrained("Salesforce/blip-base-en")
    model.to(device)
    model.eval()
    imgs_np = [images[i].cpu().permute(1, 2, 0).numpy() for i in range(images.shape[0])]
    inp = proc(images=imgs_np, return_tensors="pt")
    pixel_values = inp["pixel_values"].to(device)
    feats_orig = _subsample_layers(_collect_blip_layers(model, pixel_values, device), seed=seed)
    images_noised = _add_noise_to_images(images, epsilon, seed)
    inp_n = proc(images=[images_noised[i].cpu().permute(1, 2, 0).numpy() for i in range(images_noised.shape[0])], return_tensors="pt")
    feats_noised = _subsample_layers(_collect_blip_layers(model, inp_n["pixel_values"].to(device), device), seed=seed)
    masks = _sensitive_masks(feats_noised, True)
    return assess_privacy_budget_from_features(feats_orig, feats_noised, masks, epsilon=epsilon, bins=20, k_mdav=3)


def run(config: dict, out_dir: str) -> None:
    """
    Run VLM experiments.

    Config keys:
      - seeds: list of ints
      - epsilons: list of floats
      - num_images: int
      - size: int
      - models: list of model names, e.g. ["CLIP", "BLIP"]

    Writes vlm_metrics.csv into out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    seeds = config.get("seeds", [0, 1, 2, 3, 4])
    epsilons = config.get("epsilons", [0.1, 0.01])
    num_images = config.get("num_images", 4)
    size = config.get("size", 224)
    models = config.get("models", ["CLIP", "BLIP"])

    runners = {"CLIP": _run_clip, "BLIP": _run_blip}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    images = torch.rand(num_images, 3, size, size, device=device)

    fieldnames = ["model", "epsilon", "seed", "chi2", "kl", "mmd", "rmse", "wass1", "empa_bias_bua", "empa_bias_tda"]
    rows = []
    nan_row = lambda model, eps, s: {"model": model, "epsilon": eps, "seed": s, "chi2": np.nan, "kl": np.nan, "mmd": np.nan, "rmse": np.nan, "wass1": np.nan, "empa_bias_bua": np.nan, "empa_bias_tda": np.nan}

    for epsilon in epsilons:
        for seed in seeds:
            for model_name in models:
                if model_name not in runners:
                    continue
                try:
                    m = runners[model_name](device, images, epsilon, seed, None)
                    rows.append({"model": model_name, "epsilon": epsilon, "seed": seed, "chi2": m["chi2"], "kl": m["kl"], "mmd": m["mmd"], "rmse": m["rmse"], "wass1": m.get("wass1", np.nan), "empa_bias_bua": m["empa_bias_bua"], "empa_bias_tda": m["empa_bias_tda"]})
                    print(f"  Done {model_name} eps={epsilon} seed={seed}")
                except Exception as e:
                    print(f"  Error {model_name} eps={epsilon} seed={seed}: {e}")
                    rows.append(nan_row(model_name, epsilon, seed))

    csv_path = os.path.join(out_dir, "vlm_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")
