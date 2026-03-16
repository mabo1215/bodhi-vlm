#!/usr/bin/env python3
"""
Generate BUA-in-VLM figure for Bodhi VLM paper using fal.ai (same style as BUAexplain:
input/output, embedded example images, clear framework blocks).

Output: paper/images/BUAexplain_VLM.png (optional replacement for BUAexplain.png)

Usage:
    python -m src.scripts.generate_bua_vlm_figure [--output path/to/BUAexplain_VLM.png]

Requires: fal-client (pip install fal-client)
.env (project root): FAL_KEY=xxx

If you see SSL: CERTIFICATE_VERIFY_FAILED (e.g. corporate proxy), run with:
  set FAL_INSECURE_SSL=1
  python -m src.scripts.generate_bua_vlm_figure
or:  python -m src.scripts.generate_bua_vlm_figure --insecure-ssl
"""

import argparse
import os
import ssl
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DEFAULT = PROJECT_ROOT / "paper" / "images" / "BUAexplain_VLM.png"


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


# Same style as BUAexplain: detailed, with input/output, embedded example images, framework.
PROMPT = """Academic IEEE paper diagram, clean white background, detailed technical figure with input and output clearly shown. Same style as a flowchart that has small embedded example images and a clear pipeline.

Title: "BUA (Bottom-Up Strategy) in a VLM Vision Encoder"

Left or top: INPUT – one small example image (e.g. photo or scene) entering the pipeline, labeled "Image input".

Center: FRAMEWORK – vertical tower from bottom to top:
- Bottom: "Layer 1" = Patch embedding (or first CNN stage), label "Patch / Stage 1"
- Then "Layer 2", "Layer i", … up to "Layer n" at top (last ViT block or last ResNet stage)
- At each layer show: feature map flattened to vectors; two groups "G^(s)_i (sensitive)" and "G^(n)_i (non-sensitive)" with NCP and MDAV
- Arrows from Layer 1 upward to Layer 2, Layer i, …, Layer n. Label: "Bottom-up aggregation"
- Optional: small embedded thumbnails at one or two layers showing "sensitive vs non-sensitive" regions (e.g. highlighted patches or heatmap-style) for illustration

Right or bottom: OUTPUT – box "EMPA" (Expectation-Maximization Privacy Assessment) receiving the layer-wise partitions; then "Privacy budget bias" or "Budget assessment".

Use clear colored arrows (e.g. blue or green for bottom-up flow). No clip art, no logos. IEEE publication style. Same level of detail as a diagram that explains input, each stage, and output with embedded example visuals."""


def main():
    parser = argparse.ArgumentParser(
        description="Generate BUA-in-VLM figure via fal.ai (same style as BUAexplain)"
    )
    parser.add_argument("--output", "-o", type=Path, default=OUTPUT_DEFAULT, help="Output image path")
    parser.add_argument("--insecure-ssl", action="store_true", help="Skip SSL cert verify (e.g. behind corporate proxy)")
    args = parser.parse_args()

    if args.insecure_ssl or os.environ.get("FAL_INSECURE_SSL", "").strip().lower() in ("1", "true", "yes"):
        ssl._create_default_https_context = ssl._create_unverified_context

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
        print("To use in paper: point main.tex to this file and set caption to 'BUA in a VLM vision encoder: ...'")
    except Exception as e:
        print(f"Download error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
