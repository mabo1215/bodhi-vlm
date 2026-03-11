# -*- coding: utf-8 -*-
"""
Example integration of Bodhi VLM with an ultralytics YOLO detector.

This script demonstrates how to:
  1) Extract backbone features from YOLOv8 for a batch of images.
  2) Construct a simple sensitive mask per layer (e.g., images containing
     the COCO 'person' class are treated as sensitive).
  3) Call `assess_privacy_budget_from_features` to obtain Chi-square,
     K-L, MMD, rMSE, and EMPA bias values.

This is intended as a template; in a real system, you should replace
the dummy images and labels with your dataset and map bounding boxes
to feature-map positions if finer-grained sensitivity is desired.
"""

import numpy as np
import torch

from ultralytics import YOLO  # pip install ultralytics

from core.pipeline import assess_privacy_budget_from_features


def collect_yolo_backbone_features(model: YOLO, images: torch.Tensor) -> list[np.ndarray]:
    """
    Run a forward pass through YOLO and collect intermediate backbone features.

    Parameters
    ----------
    model:
        An ultralytics YOLO model (e.g., YOLO('yolov8n.pt')).
    images:
        Input batch of images, shape (B, 3, H, W).

    Returns
    -------
    feats_per_layer:
        List of arrays, one per tapped layer, each of shape (N_i, D) where
        N_i = B * H_i * W_i and D = channel dimension.
    """
    feats_per_layer: list[np.ndarray] = []
    handles = []

    def make_hook():
        def hook(m, inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            b, c, h, w = x.shape
            feat = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
            feats_per_layer.append(feat.detach().cpu().numpy())
        return hook

    # Example: hook several backbone convolutional blocks.
    for name, module in model.model.named_modules():
        if "backbone" in name and any(k in name for k in ["cv2", "cv3", "cv4"]):
            handles.append(module.register_forward_hook(make_hook()))

    with torch.no_grad():
        _ = model(images)

    for h in handles:
        h.remove()

    return feats_per_layer


def build_sensitive_masks_for_yolo(
    layer_features: list[np.ndarray],
    has_sensitive: bool,
) -> list[np.ndarray]:
    """
    Simplified sensitive mask: if the image contains any sensitive label,
    mark all spatial positions in all layers as sensitive.
    """
    masks: list[np.ndarray] = []
    for feat in layer_features:
        n = feat.shape[0]
        if has_sensitive:
            mask = np.ones(n, dtype=bool)
        else:
            mask = np.zeros(n, dtype=bool)
        masks.append(mask)
    return masks


def add_gaussian_noise_to_images(images: torch.Tensor, epsilon: float = 0.1, scale: float = 1.0) -> torch.Tensor:
    """
    Add Gaussian privacy noise to input images (toy example).
    """
    sigma = scale / (epsilon + 1e-8)
    noise = torch.randn_like(images) * sigma
    return torch.clamp(images + noise, 0.0, 1.0)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    model.to(device)
    model.eval()

    # Dummy input; replace with your preprocessed images and labels.
    img = torch.rand(1, 3, 640, 640, device=device)

    # Assume the image contains a sensitive class (e.g., 'person' in COCO).
    has_sensitive = True

    # Original features
    feats_orig = collect_yolo_backbone_features(model, img)

    # Noised features
    eps = 0.1
    img_noised = add_gaussian_noise_to_images(img, epsilon=eps)
    feats_noised = collect_yolo_backbone_features(model, img_noised)

    # Sensitive masks per layer
    sensitive_masks = build_sensitive_masks_for_yolo(feats_noised, has_sensitive=has_sensitive)

    metrics = assess_privacy_budget_from_features(
        layer_features_orig=feats_orig,
        layer_features_noised=feats_noised,
        sensitive_per_layer=sensitive_masks,
        epsilon=eps,
        bins=20,
        k_mdav=3,
    )

    print("Bodhi VLM metrics on YOLO backbone:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

