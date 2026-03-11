# -*- coding: utf-8 -*-
"""
Example integration of Bodhi VLM with the OpenAI CLIP ViT vision encoder.

This script demonstrates how to:
  1) Extract layer-wise patch features from CLIP's visual encoder.
  2) Construct a simple sensitive mask per image (e.g., whether the image
     is considered sensitive based on metadata or labels).
  3) Call `assess_privacy_budget_from_features` to obtain the same metrics
     used in the Bodhi VLM paper.

Dependencies:
  pip install git+https://github.com/openai/CLIP.git
"""

import numpy as np
import torch

import clip  # type: ignore

from core.pipeline import assess_privacy_budget_from_features


def collect_clip_vit_layers(model, images: torch.Tensor) -> list[np.ndarray]:
    """
    Collect layer-wise patch features from the CLIP ViT visual encoder.

    Parameters
    ----------
    model:
        CLIP model as returned by `clip.load("ViT-B/32")`.
    images:
        Batch of images, shape (B, 3, H, W), preprocessed using
        the CLIP `preprocess` function.
    """
    feats_per_layer: list[np.ndarray] = []
    handles = []
    visual = model.visual

    def make_hook():
        def hook(m, inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            # x: (B, 1+P, D) — CLS + patch tokens
            x = x[:, 1:, :]
            b, p, d = x.shape
            feats_per_layer.append(x.reshape(b * p, d).detach().cpu().numpy())
        return hook

    for block in visual.transformer.resblocks:
        handles.append(block.register_forward_hook(make_hook()))

    with torch.no_grad():
        _ = model.encode_image(images)

    for h in handles:
        h.remove()

    return feats_per_layer


def build_sensitive_masks_for_clip(
    layer_features: list[np.ndarray],
    has_sensitive_per_image: list[bool],
) -> list[np.ndarray]:
    """
    Simple sensitive mask: if an image is marked as sensitive,
    all of its patch tokens are marked sensitive.
    """
    B = len(has_sensitive_per_image)
    masks: list[np.ndarray] = []
    for feat in layer_features:
        n, d = feat.shape
        patches_per_image = n // B
        mask = np.zeros(n, dtype=bool)
        for i, flag in enumerate(has_sensitive_per_image):
            if flag:
                start = i * patches_per_image
                end = (i + 1) * patches_per_image
                mask[start:end] = True
        masks.append(mask)
    return masks


def add_laplace_noise_to_images(images: torch.Tensor, epsilon: float = 0.1, scale: float = 1.0) -> torch.Tensor:
    """
    Add Laplace noise to input images (toy example).
    """
    b, c, h, w = images.shape
    b_param = scale / (epsilon + 1e-8)
    u = torch.rand_like(images) - 0.5
    noise = -b_param * torch.sign(u) * torch.log1p(-2 * torch.abs(u))
    return torch.clamp(images + noise, 0.0, 1.0)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Dummy images; replace with real images preprocessed by `preprocess`.
    img = torch.rand(2, 3, 224, 224, device=device)
    img_pre = img

    # Example: mark the first image as sensitive, the second as non-sensitive.
    has_sensitive_per_image = [True, False]

    feats_orig = collect_clip_vit_layers(model, img_pre)

    eps = 0.1
    img_noised = add_laplace_noise_to_images(img_pre, epsilon=eps)
    feats_noised = collect_clip_vit_layers(model, img_noised)

    sensitive_masks = build_sensitive_masks_for_clip(
        layer_features=feats_noised,
        has_sensitive_per_image=has_sensitive_per_image,
    )

    metrics = assess_privacy_budget_from_features(
        layer_features_orig=feats_orig,
        layer_features_noised=feats_noised,
        sensitive_per_layer=sensitive_masks,
        epsilon=eps,
        bins=20,
        k_mdav=3,
    )

    print("Bodhi VLM metrics on CLIP ViT-B/32 vision encoder:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

