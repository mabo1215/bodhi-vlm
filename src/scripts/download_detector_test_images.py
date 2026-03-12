#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download test images for detector experiments (YOLO/DETR) so runs yield real detections.
Uses detection-friendly URLs (people, cat, car, dog, etc.). Saves to data/test_images/.
Run from repo root: python src/scripts/download_detector_test_images.py [N]
To refresh images: remove data/test_images/*.jpg then run again.
"""
import os
import sys
import urllib.request

# Project root (parent of src/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

# Default: 6 images so we have enough for num_images=4 with variety
DEFAULT_NUM = 6
# Picsum.photos fixed IDs (each id = one image; no rate limit). Mix of people, animals, objects.
# You can replace IDs: https://picsum.photos/ lists many (e.g. 237=dog, 659=person, 1025=person).
DETECTION_FRIENDLY_URLS = [
    "https://picsum.photos/id/237/640/480",   # dog
    "https://picsum.photos/id/659/640/480",   # person
    "https://picsum.photos/id/1025/640/480",  # person/couple
    "https://picsum.photos/id/10/640/480",   # mountain
    "https://picsum.photos/id/1018/640/480",  # building
    "https://picsum.photos/id/1074/640/480",  # city/snow
]
FALLBACK_URLS = [
    "https://picsum.photos/seed/99/640/480",
    "https://picsum.photos/seed/100/640/480",
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
    out_dir = os.path.join(_PROJECT_ROOT, "data", "test_images")
    os.makedirs(out_dir, exist_ok=True)

    num = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NUM
    print(f"Downloading up to {num} test images (detection-friendly) to {out_dir}")

    saved = 0
    # Prefer curated URLs that tend to contain person/car/cat for YOLO/COCO
    for i in range(min(num, len(DETECTION_FRIENDLY_URLS))):
        dest = os.path.join(out_dir, f"test_{i + 1:02d}.jpg")
        if os.path.exists(dest):
            print(f"  Keep existing {os.path.basename(dest)}")
            saved += 1
            continue
        if download_one(DETECTION_FRIENDLY_URLS[i], dest):
            print(f"  Saved {os.path.basename(dest)}")
            saved += 1

    # Fill remaining slots with fallbacks
    next_slot = saved + 1
    for url in FALLBACK_URLS:
        if saved >= num:
            break
        dest = os.path.join(out_dir, f"test_{next_slot:02d}.jpg")
        if not os.path.exists(dest) and download_one(url, dest):
            print(f"  Saved {os.path.basename(dest)}")
            saved += 1
        next_slot += 1

    print(f"Done. {saved} images in {out_dir}")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
