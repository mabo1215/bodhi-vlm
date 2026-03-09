# -*- coding: utf-8 -*-
"""
Read bodhi_vlm_metrics.csv and generate paper figures into ../images/
Run from scripts/: python plot_paper_figures.py
"""
import os
import csv

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required: pip install matplotlib")
    exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
IMAGES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "images")
CSV_PATH = os.path.join(RESULTS_DIR, "bodhi_vlm_metrics.csv")

def load_csv():
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def main():
    if not os.path.isfile(CSV_PATH):
        print(f"Run experiments first: {CSV_PATH} not found")
        return
    os.makedirs(IMAGES_DIR, exist_ok=True)
    rows = load_csv()
    eps_list = [r["epsilon"] for r in rows]

    # Figure 1: EMPA bias vs epsilon
    fig, ax = plt.subplots(1, 1, figsize=(4, 2.8))
    ax.plot(eps_list, [r["empa_bias_bua"] for r in rows], "o-", label="BUA+EMPA bias", markersize=8)
    ax.plot(eps_list, [r["empa_bias_tda"] for r in rows], "s-", label="TDA+EMPA bias", markersize=8)
    ax.set_xlabel("Privacy budget $\\epsilon$")
    ax.set_ylabel("EMPA bias")
    ax.legend()
    ax.set_title("Bodhi VLM: EMPA bias vs budget")
    ax.set_xscale("log")
    fig.tight_layout()
    out1 = os.path.join(IMAGES_DIR, "bodhi_vlm_empa_bias.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")

    # Figure 2: rMSE and metrics vs epsilon
    fig2, axes = plt.subplots(1, 2, figsize=(6, 2.6))
    ax1, ax2 = axes[0], axes[1]
    ax1.semilogy(eps_list, [r["rmse"] for r in rows], "o-", color="C0", label="rMSE", markersize=8)
    ax1.set_xlabel("Privacy budget $\\epsilon$")
    ax1.set_ylabel("rMSE")
    ax1.set_title("rMSE vs $\\epsilon$")
    ax1.set_xscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.semilogy(eps_list, [r["chi2"] for r in rows], "^-", color="C1", label="Chi-square", markersize=6)
    ax2.semilogy(eps_list, [r["kl"] for r in rows], "s-", color="C2", label="K-L div.", markersize=6)
    ax2.semilogy(eps_list, [r["mmd"] for r in rows], "d-", color="C3", label="MMD", markersize=6)
    ax2.set_xlabel("Privacy budget $\\epsilon$")
    ax2.set_ylabel("Metric value")
    ax2.set_title("Baseline metrics vs $\\epsilon$")
    ax2.set_xscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(IMAGES_DIR, "bodhi_vlm_metrics_vs_epsilon.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

if __name__ == "__main__":
    main()
