# -*- coding: utf-8 -*-
"""Bodhi VLM experiment entrypoints driven by config.json and main.py.
Lazy-import detector/vlm/interpretability so running only synthetic does not require torch."""
from experiments.synthetic import run as run_synthetic
from experiments.aggregate import run as run_aggregate

def __getattr__(name):
    if name == "run_detector":
        from experiments.detector import run as run_detector
        return run_detector
    if name == "run_vlm":
        from experiments.vlm import run as run_vlm
        return run_vlm
    if name == "run_interpretability":
        from experiments.interpretability import run as run_interpretability
        return run_interpretability
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "run_synthetic",
    "run_detector",
    "run_vlm",
    "run_aggregate",
    "run_interpretability",
]
