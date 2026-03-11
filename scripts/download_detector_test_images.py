#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download test images for detector experiments (YOLO/DETR) so runs yield real detections.
Saves images under data/test_images/. Run from repo root: python scripts/download_detector_test_images.py
"""
import os
import sys
import urllib.request

# Default: 6 images so we have enough for num_images=4 with variety
DEFAULT_NUM = 6
# Deterministic "random" images from picsum.photos (real photos, may contain people/objects)
PICSUM_TEMPLATE = "https://picsum.photos/seed/{seed}/640/480"
# Fallback: single sample from a stable CDN
FALLBACK_URLS = [
    "https://placekitten.com/640/480",
    "https://picsum.photos/640/480",
]


def download_one(url: str, dest: str) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Bodhi-VLM/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"  Skip {url}: {e}")
        return False


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "data", "test_images")
    os.makedirs(out_dir, exist_ok=True)

    num = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NUM
    print(f"Downloading up to {num} test images to {out_dir}")

    saved = 0
    for i in range(num):
        url = PICSUM_TEMPLATE.format(seed=i + 1)
        dest = os.path.join(out_dir, f"test_{i + 1:02d}.jpg")
        if os.path.exists(dest):
            print(f"  Keep existing {os.path.basename(dest)}")
            saved += 1
            continue
        if download_one(url, dest):
            print(f"  Saved {os.path.basename(dest)}")
            saved += 1

    for i, url in enumerate(FALLBACK_URLS):
        if saved >= num:
            break
        dest = os.path.join(out_dir, f"fallback_{i + 1}.jpg")
        if os.path.exists(dest) or download_one(url, dest):
            saved += 1

    print(f"Done. {saved} images in {out_dir}")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
