#!/usr/bin/env python3
"""
Generate MMDEMPA.png: Deviation comparison of TDA+EMPA vs BUA+EMPA
for Bodhi VLM paper using fal.ai Nano Banana 2 API.

Usage:
    python generate_mmdempa_figure.py [--output path/to/MMDEMPA.png]

Requires:
    pip install fal-client python-dotenv
    # Or: pip install -r requirements_generate_figure.txt

.env (project root) should contain FAL_KEY=xxx or raw API key on first line.
Get API key: https://fal.ai/dashboard/keys
"""

import argparse
import os
import sys
from pathlib import Path

# Load .env from project root (one level up from paper/scripts)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DEFAULT = PROJECT_ROOT / "paper" / "images" / "MMDEMPA.png"


def load_fal_key() -> str:
    """Load FAL_KEY from .env file."""
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
                    # Raw key (entire line is FAL_KEY)
                    return line.strip()
    return os.environ.get("FAL_KEY", "")


PROMPT = """Academic IEEE paper flowchart, clean white background, no decorative elements, formal technical diagram.

Title at top: "TDA+EMPA vs BUA+EMPA - Privacy Budget Assessment"

Left column - BUA+EMPA (Bottom-Up Analysis):
- Start: "Layer 1 (bottom)" 
- Arrow pointing UP through "Layer 2" → "Layer i" → "Layer n (top)"
- Label: "Bottom-up aggregation: G_1,...,G_n"
- Box: "EMPA" with sub-labels "E-step" and "M-step"
- Output: "Bias (BUA)"

Right column - TDA+EMPA (Top-Down Analysis):
- Start: "Layer n (top)"
- Arrow pointing DOWN through "Layer i" → "Layer 2" → "Layer 1 (bottom)"
- Label: "Top-down partition: G_i, G'_i"
- Box: "EMPA" with sub-labels "E-step" and "M-step"
- Output: "Bias (TDA)"

Center - Comparison:
- "Deviation = |Bias(TDA) - Bias(BUA)|"
- Small plot sketch: x-axis "Image index", y-axis "Deviation", two overlapping curves

Minimalist, black lines on white, IEEE publication style, no icons or clutter."""


def main():
    parser = argparse.ArgumentParser(description="Generate TDA+EMPA vs BUA+EMPA figure via fal.ai")
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
