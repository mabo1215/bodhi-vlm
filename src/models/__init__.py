# -*- coding: utf-8 -*-
"""Central model cache: all downloaded models (HF, torch.hub, *.pt) under src/models/."""
import os

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = _MODELS_DIR
HF_CACHE = os.path.join(_MODELS_DIR, "huggingface")
TORCH_HUB_DIR = os.path.join(_MODELS_DIR, "torch_hub")
WEIGHTS_DIR = os.path.join(_MODELS_DIR, "weights")


def setup_models_dir() -> None:
    """Create model subdirs and set env so Hugging Face and torch.hub use them."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HF_CACHE, exist_ok=True)
    os.makedirs(TORCH_HUB_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
    try:
        import torch
        torch.hub.set_dir(TORCH_HUB_DIR)
    except ImportError:
        pass


def get_weights_path(filename: str) -> str:
    """Return path under src/models/weights/; ensure weights dir exists."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    return os.path.join(WEIGHTS_DIR, filename)


def ensure_yolo_weights(filename: str = "yolov8n.pt") -> str:
    """Download YOLO weights into src/models/weights/ if not present; return path."""
    path = get_weights_path(filename)
    if os.path.exists(path):
        return path
    old_cwd = os.getcwd()
    try:
        os.chdir(WEIGHTS_DIR)
        from ultralytics import YOLO
        YOLO(filename)
    finally:
        os.chdir(old_cwd)
    return path
