# -*- coding: utf-8 -*-
"""Aggregate multi-seed metrics into mean±std and emit LaTeX tables."""
import os
import pandas as pd
from typing import List


def agg_mean_std(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    agg_dict = {f"{c}_mean": (c, "mean") for c in value_cols}
    agg_dict.update({f"{c}_std": (c, "std") for c in value_cols})
    return df.groupby(group_cols).agg(**agg_dict).reset_index()


def format_pm(mean: float, std: float, precision: int = 3) -> str:
    if pd.isna(mean):
        return "---"
    if pd.isna(std):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def detector_latex_table(summary: pd.DataFrame) -> str:
    summary = summary.set_index(["model", "epsilon"])

    def cell(model: str, eps: float, metric: str) -> str:
        try:
            row = summary.loc[(model, eps)]
            return format_pm(row[f"{metric}_mean"], row[f"{metric}_std"], precision=2)
        except (KeyError, TypeError):
            return "---"

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{rMSE (mean $\\pm$ std over $R$ seeds) for Chi-square, K-L, MMD, EMPA on detector backbones.}",
        "\\label{tab:rmse}",
        "\\begin{tabular}{l|cccc}",
        "\\hline",
        "Indicator & MDCRF-0.1 & DETR-0.1 & PPDPTS-0.1 & PPDPTS-0.01 \\\\",
        "\\hline",
        f"Chi-square & {cell('MDCRF', 0.1, 'rmse')} & {cell('DETR', 0.1, 'rmse')} & {cell('PPDPTS', 0.1, 'rmse')} & {cell('PPDPTS', 0.01, 'rmse')} \\\\",
        f"K-L Div. & {cell('MDCRF', 0.1, 'kl')} & {cell('DETR', 0.1, 'kl')} & {cell('PPDPTS', 0.1, 'kl')} & {cell('PPDPTS', 0.01, 'kl')} \\\\",
        f"MMD & {cell('MDCRF', 0.1, 'mmd')} & {cell('DETR', 0.1, 'mmd')} & {cell('PPDPTS', 0.1, 'mmd')} & {cell('PPDPTS', 0.01, 'mmd')} \\\\",
        f"EMPA & {cell('MDCRF', 0.1, 'empa_bias_bua')} & {cell('DETR', 0.1, 'empa_bias_bua')} & {cell('PPDPTS', 0.1, 'empa_bias_bua')} & {cell('PPDPTS', 0.01, 'empa_bias_bua')} \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def vlm_latex_table(summary: pd.DataFrame) -> str:
    if "model" not in summary.columns or "epsilon" not in summary.columns:
        return "% VLM summary missing model/epsilon"
    summary = summary.set_index(["model", "epsilon"])

    def cell(model: str, eps: float, metric: str) -> str:
        try:
            row = summary.loc[(model, eps)]
            return format_pm(row[f"{metric}_mean"], row[f"{metric}_std"], precision=2)
        except Exception:
            return "---"

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{VLM vision encoder metrics (mean $\\pm$ std over seeds).}",
        "\\label{tab:vlmresults}",
        "\\begin{tabular}{l|cc}",
        "\\hline",
        "Model & rMSE & EMPA (BUA) \\\\",
        "\\hline",
    ]
    for model in summary.index.get_level_values("model").unique():
        for eps in summary.index.get_level_values("epsilon").unique():
            lines.append(f"{model} $\\varepsilon={eps}$ & {cell(model, eps, 'rmse')} & {cell(model, eps, 'empa_bias_bua')} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def run(config: dict, out_dir: str) -> None:
    """
    Run aggregation.

    Config keys:
      - detector_csv: path to detector_metrics.csv (relative to out_dir if not absolute)
      - vlm_csv: path to vlm_metrics.csv (relative to out_dir if not absolute)

    Writes LaTeX tables into out_dir/detector_tab_rmse.tex and/or out_dir/vlm_tab_results.tex.
    """
    os.makedirs(out_dir, exist_ok=True)
    def resolve(p: str) -> str:
        if not p or os.path.isabs(p):
            return p
        return os.path.join(out_dir, p) if not p.startswith("..") else os.path.abspath(p)
    detector_csv = resolve(config.get("detector_csv", ""))
    vlm_csv = resolve(config.get("vlm_csv", ""))
    if detector_csv and os.path.exists(detector_csv):
        df = pd.read_csv(detector_csv)
        summary = agg_mean_std(df, ["model", "epsilon"], ["rmse", "kl", "mmd", "empa_bias_bua"])
        tex = detector_latex_table(summary)
        path = os.path.join(out_dir, "detector_tab_rmse.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"Wrote {path}")
    if vlm_csv and os.path.exists(vlm_csv):
        df = pd.read_csv(vlm_csv)
        summary = agg_mean_std(df, ["model", "epsilon"], ["rmse", "kl", "mmd", "empa_bias_bua"])
        tex = vlm_latex_table(summary)
        path = os.path.join(out_dir, "vlm_tab_results.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"Wrote {path}")
