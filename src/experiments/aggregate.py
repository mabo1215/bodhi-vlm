# -*- coding: utf-8 -*-
"""Aggregate multi-seed metrics into mean±std, 95%% CI, budget ranking; emit LaTeX tables."""
import os
import pandas as pd
import numpy as np
from typing import List

try:
    from utils.metrics import confidence_interval_95, budget_ranking_correct_two
except ImportError:
    confidence_interval_95 = None
    budget_ranking_correct_two = None


def agg_mean_std(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    agg_dict = {f"{c}_mean": (c, "mean") for c in value_cols}
    agg_dict.update({f"{c}_std": (c, "std") for c in value_cols})
    return df.groupby(group_cols).agg(**agg_dict).reset_index()


def agg_mean_std_ci(
    df: pd.DataFrame,
    group_cols: List[str],
    value_cols: List[str],
) -> pd.DataFrame:
    """Like agg_mean_std but add ci_low, ci_high (95%%) per value column."""
    parts = []
    for key, grp in df.groupby(group_cols):
        key_tup = key if isinstance(key, tuple) else (key,)
        row = {g: key_tup[i] for i, g in enumerate(group_cols)}
        for c in value_cols:
            vals = grp[c].dropna().values
            row[f"{c}_mean"] = np.mean(vals)
            row[f"{c}_std"] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            if confidence_interval_95 is not None and len(vals) >= 2:
                lo, hi = confidence_interval_95(vals)
                row[f"{c}_ci_low"] = lo
                row[f"{c}_ci_high"] = hi
            else:
                row[f"{c}_ci_low"] = row[f"{c}_mean"]
                row[f"{c}_ci_high"] = row[f"{c}_mean"]
        parts.append(row)
    return pd.DataFrame(parts)


def budget_ranking_from_df(
    df: pd.DataFrame,
    group_cols: List[str],
    epsilons: List[float],
    score_col: str = "empa_bias_bua",
    lower_is_better: bool = False,
) -> pd.DataFrame:
    """
    For each group (e.g. model), compute fraction of seeds where rank(scores) matches rank(epsilons).
    df must have columns: group_cols + epsilon + seed + score_col.
    """
    if budget_ranking_correct_two is None or len(epsilons) != 2:
        return pd.DataFrame()
    eps_small, eps_large = min(epsilons), max(epsilons)
    groups = df.groupby(group_cols)
    rows = []
    for name, grp in groups:
        if isinstance(name, (list, tuple)):
            key = dict(zip(group_cols, name))
        else:
            key = {group_cols[0]: name}
        by_seed = grp.groupby("seed")
        correct = 0
        total = 0
        for seed, sd in by_seed:
            if sd.shape[0] < 2:
                continue
            eps_vals = sd["epsilon"].values
            sc_vals = sd[score_col].values
            if len(eps_vals) != 2 or np.any(np.isnan(sc_vals)):
                continue
            i_small = np.argmin(eps_vals)
            i_large = 1 - i_small
            ok = budget_ranking_correct_two(
                eps_vals[i_small], eps_vals[i_large],
                sc_vals[i_small], sc_vals[i_large],
                lower_is_better,
            )
            correct += 1 if ok else 0
            total += 1
        acc = correct / total if total else np.nan
        rows.append({**key, "budget_ranking_accuracy": acc, "n_seeds": total})
    return pd.DataFrame(rows)


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
        value_cols = ["chi2", "kl", "mmd", "rmse", "wass1", "empa_bias_bua", "empa_bias_tda"]
        value_cols = [c for c in value_cols if c in df.columns]
        summary = agg_mean_std(df, ["model", "epsilon"], value_cols)
        summary_ci = agg_mean_std_ci(df, ["model", "epsilon"], value_cols)
        summary_ci.to_csv(os.path.join(out_dir, "detector_summary_with_ci.csv"), index=False)
        epsilons = sorted(df["epsilon"].dropna().unique().tolist())
        if len(epsilons) >= 2 and budget_ranking_correct_two is not None:
            br = budget_ranking_from_df(df, ["model"], epsilons, score_col="empa_bias_bua", lower_is_better=False)
            br.to_csv(os.path.join(out_dir, "detector_budget_ranking.csv"), index=False)
            print("  detector: budget_ranking_accuracy (empa_bias_bua):", br["budget_ranking_accuracy"].tolist())
        tex = detector_latex_table(summary)
        path = os.path.join(out_dir, "detector_tab_rmse.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"Wrote {path}, detector_summary_with_ci.csv")
    if vlm_csv and os.path.exists(vlm_csv):
        df = pd.read_csv(vlm_csv)
        value_cols = ["chi2", "kl", "mmd", "rmse", "wass1", "empa_bias_bua", "empa_bias_tda"]
        value_cols = [c for c in value_cols if c in df.columns]
        summary = agg_mean_std(df, ["model", "epsilon"], value_cols)
        summary_ci = agg_mean_std_ci(df, ["model", "epsilon"], value_cols)
        summary_ci.to_csv(os.path.join(out_dir, "vlm_summary_with_ci.csv"), index=False)
        epsilons = sorted(df["epsilon"].dropna().unique().tolist())
        if len(epsilons) >= 2 and budget_ranking_correct_two is not None:
            br = budget_ranking_from_df(df, ["model"], epsilons, score_col="empa_bias_bua", lower_is_better=False)
            br.to_csv(os.path.join(out_dir, "vlm_budget_ranking.csv"), index=False)
            print("  vlm: budget_ranking_accuracy (empa_bias_bua):", br["budget_ranking_accuracy"].tolist())
        tex = vlm_latex_table(summary)
        path = os.path.join(out_dir, "vlm_tab_results.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"Wrote {path}, vlm_summary_with_ci.csv")
