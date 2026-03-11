#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Write Hugging Face token to .env so gated models (e.g. BLIP) can be loaded.
Run from repo root: python scripts/setup_hf_token.py
Or: python scripts/setup_hf_token.py hf_your_token_here
"""
import os
import sys

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(repo_root, ".env")

    if len(sys.argv) > 1:
        token = sys.argv[1].strip()
    else:
        print("Paste your Hugging Face token (from https://huggingface.co/settings/tokens):")
        token = (sys.stdin.readline() or "").strip()

    if not token:
        print("No token provided. Create .env manually with: HF_TOKEN=hf_...")
        example = os.path.join(repo_root, ".env.example")
        if os.path.exists(example):
            print(f"See {example} for format.")
        return 1

    line = f"HF_TOKEN={token}\n"
    with open(env_path, "w") as f:
        f.write("# Hugging Face token for gated models\n")
        f.write(line)
    print(f"Wrote HF_TOKEN to {env_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
