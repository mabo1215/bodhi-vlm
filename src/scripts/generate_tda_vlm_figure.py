#!/usr/bin/env python3
"""
Generate PPDPTSexplain.png: TDA and BUA in Vision-Language Models (VLM)
for Bodhi VLM paper using fal.ai Nano Banana 2 API.

Content: Top-Down Analysis (TDA) and Bottom-Up Analysis (BUA) over VLM
vision encoder layers; no PPDPTS; emphasis on VLM integration.
Style: Clean white background, no irrelevant icons, color allowed.

Usage:
    python generate_tda_vlm_figure.py [--output path/to/PPDPTSexplain.png]

Requires: fal-client (pip install fal-client)
.env (project root): FAL_KEY=xxx or raw API key on first line.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DEFAULT = PROJECT_ROOT / "paper" / "images" / "PPDPTSexplain.png"


def load_fal_key() -> str:
    if DOTENV_PATH.exists():
        with open(DOTENV_PATH) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    if key.strip().upper() in ("FAL_KEY", "FAL_CREDENTIALS"):
                        return val.strip().strip('"').strip("'")
                else:
                    return line.strip()
    return os.environ.get("FAL_KEY", "")


PROMPT = """Academic IEEE paper diagram, clean white background, no decorative icons. Colorful arrows and boxes allowed. Formal technical figure.

Title: "TDA and BUA in Vision-Language Models (VLM)"

Central block: "VLM Vision Encoder" (e.g. ViT or ResNet) with "Layer 1" at bottom and "Layer n" at top. Show stacked layers as a vertical tower.

Left side - BUA (Bottom-Up Analysis):
- Arrow from "Layer 1 (bottom)" upward through "Layer 2", "Layer i", to "Layer n (top)"
- Label: "Bottom-up aggregation"
- At each layer: "G_i (non-sensitive), G'_i (sensitive)" with NCP and MDAV
- Flow into box "EMPA" then "Privacy budget bias"

Right side - TDA (Top-Down Analysis):
- Arrow from "Layer n (top)" downward through "Layer i", "Layer 2", to "Layer 1 (bottom)"
- Label: "Top-down partition"
- At each layer: "G_i, G'_i" with MDAV and NCP
- Flow into box "EMPA" then "Privacy budget bias"

Bottom: "Image input" into vision encoder; "Layer-wise features" feed both BUA and TDA. Sensitive set S_epsilon.

Use clear colored arrows (e.g. blue for BUA, green for TDA). No clip art, no logos. IEEE publication style."""


def main():
    parser = argparse.ArgumentParser(description="Generate TDA/BUA in VLM figure via fal.ai")
    parser.add_argument("--output", "-o", type=Path, default=OUTPUT_DEFAULT, help="Output image path")
    args = parser.parse_args()

    key = load_fal_key()
    if not key:
        print("FAL_KEY not found. Set FAL_KEY in .env or environment.", file=sys.stderr)
        sys.exit(1)

    os.environ["FAL_KEY"] = key

    try:
        import fal_client
    except ImportError:
        print("Install fal-client: pip install fal-client", file=sys.stderr)
        sys.exit(1)

    try:
        result = fal_client.subscribe(
            "fal-ai/nano-banana-2",
            arguments={
                "prompt": PROMPT,
                "aspect_ratio": "16:9",
                "resolution": "1K",
                "output_format": "png",
            },
        )
    except Exception as e:
        print(f"API error: {e}", file=sys.stderr)
        sys.exit(1)

    images = result.get("images") or result.get("data", {}).get("images", [])
    if not images:
        print("No images in response:", result, file=sys.stderr)
        sys.exit(1)

    url = images[0].get("url")
    if not url:
        print("No URL in image:", images[0], file=sys.stderr)
        sys.exit(1)

    try:
        import urllib.request
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, output_path)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Download error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
