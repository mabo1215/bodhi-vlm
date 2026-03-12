#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copy images in data/test_images/ to test_01.jpg ... test_10.jpg (originals unchanged).
Run from repo root: python src/scripts/copy_test_images_as_named.py
"""
import os
import glob
import shutil

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
TEST_IMAGES_DIR = os.path.join(_PROJECT_ROOT, "data", "test_images")
DEFAULT_N = 10


def main():
    if not os.path.isdir(TEST_IMAGES_DIR):
        print(f"Directory not found: {TEST_IMAGES_DIR}")
        return 1
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, ext)))
    # Exclude already-named test_XX so we sort only "originals"
    paths = [p for p in paths if not os.path.basename(p).lower().startswith("test_")]
    paths = sorted(paths)[:DEFAULT_N]
    if len(paths) < DEFAULT_N:
        print(f"Found {len(paths)} images (need {DEFAULT_N}). Add more to {TEST_IMAGES_DIR}")
        return 1
    for i, src in enumerate(paths, start=1):
        dest = os.path.join(TEST_IMAGES_DIR, f"test_{i:02d}.jpg")
        if os.path.normpath(src) != os.path.normpath(dest):
            shutil.copy2(src, dest)
            print(f"  {os.path.basename(src)} -> test_{i:02d}.jpg")
    print(f"Done. {DEFAULT_N} copies as test_01.jpg ... test_{DEFAULT_N:02d}.jpg (originals kept).")
    return 0


if __name__ == "__main__":
    exit(main())
