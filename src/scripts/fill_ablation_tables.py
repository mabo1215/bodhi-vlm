# -*- coding: utf-8 -*-
"""
Read results/ablation_component_summary.csv and write LaTeX for Table S3 (supplementary)
and for main.tex Table~\\ref{tab:ablation}. Uses same data for PPDPTS and COCO columns
(detector/DETR protocol). Run after run_detector_ablation_only.py or detector with run_component_ablation.
"""
import os
import csv
import argparse

def _float(x):
    if x is None or x == "" or str(x).strip().lower() == "nan":
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None

def _cell(mean, std, ci_low, ci_high, with_ci=True):
    if mean is None or (std is not None and std != std) or (mean != mean):
        return "---"
    m, s = round(mean, 2), round(std, 2) if std is not None else 0
    if with_ci and ci_low is not None and ci_high is not None and ci_low == ci_low and ci_high == ci_high:
        return f"${m} \\pm {s}$ ({ci_low:.2f}--{ci_high:.2f})"
    return f"${m} \\pm {s}$"


def _cell_main(mean, std, bold=False):
    """Main table: mean$\\pm$std (no CI). bold for full-pipeline rMSE."""
    if mean is None or (std is not None and std != std) or (mean != mean):
        return "---"
    m, s = round(mean, 2), round(std, 2) if std is not None else 0
    cell = f"{m}$\\pm${s}"
    return f"\\textbf{{{cell}}}" if bold else cell

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--patch", action="store_true", help="Patch main.tex and supplementary.tex with generated table bodies")
    args = parser.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = args.out_dir or os.path.join(root, "results")
    csv_path = args.summary_csv or os.path.join(out_dir, "ablation_component_summary.csv")
    if not os.path.isfile(csv_path):
        print(f"Not found: {csv_path}")
        return
    by_key = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cfg, eps = row["config"], float(row["epsilon"])
            by_key[(cfg, eps)] = {
                "dev_mean": _float(row.get("dev_mean")),
                "dev_std": _float(row.get("dev_std")),
                "dev_ci_low": _float(row.get("dev_ci_low")),
                "dev_ci_high": _float(row.get("dev_ci_high")),
                "rmse_mean": _float(row.get("rmse_mean")),
                "rmse_std": _float(row.get("rmse_std")),
                "rmse_ci_low": _float(row.get("rmse_ci_low")),
                "rmse_ci_high": _float(row.get("rmse_ci_high")),
            }
    # Row order: full, tda_only (w/o BUA), bua_only (w/o TDA), no_empa
    row_configs = [("full", "Bodhi VLM (full)"), ("tda_only", "w/o BUA (TDA only)"), ("bua_only", "w/o TDA (BUA only)"), ("no_empa", "w/o EMPA")]
    epsilons = [0.1, 0.01]
    def get(cfg, eps, field):
        d = by_key.get((cfg, eps), {})
        return d.get(field)

    # Build main table body (mean +/- std only; bold rMSE for full)
    main_lines = []
    for cfg, label in row_configs:
        cells = [label]
        for eps in epsilons:
            m, s = get(cfg, eps, "dev_mean"), get(cfg, eps, "dev_std")
            cells.append(_cell_main(m, s, bold=False))
        for eps in epsilons:
            m, s = get(cfg, eps, "dev_mean"), get(cfg, eps, "dev_std")
            cells.append(_cell_main(m, s, bold=False))
        for eps in epsilons:
            m, s = get(cfg, eps, "rmse_mean"), get(cfg, eps, "rmse_std")
            cells.append(_cell_main(m, s, bold=(cfg == "full")))
        main_lines.append(" & ".join(cells) + " \\\\")
    main_body = "\n".join(main_lines)

    # Build supplementary table body (mean +/- std and 95% CI)
    sup_lines = []
    for cfg, label in row_configs:
        cells = [label]
        for eps in epsilons:
            m, s = get(cfg, eps, "dev_mean"), get(cfg, eps, "dev_std")
            lo, hi = get(cfg, eps, "dev_ci_low"), get(cfg, eps, "dev_ci_high")
            cells.append(_cell(m, s, lo, hi, with_ci=True))
        for eps in epsilons:
            m, s = get(cfg, eps, "dev_mean"), get(cfg, eps, "dev_std")
            lo, hi = get(cfg, eps, "dev_ci_low"), get(cfg, eps, "dev_ci_high")
            cells.append(_cell(m, s, lo, hi, with_ci=True))
        for eps in epsilons:
            m, s = get(cfg, eps, "rmse_mean"), get(cfg, eps, "rmse_std")
            lo, hi = get(cfg, eps, "rmse_ci_low"), get(cfg, eps, "rmse_ci_high")
            cells.append(_cell(m, s, lo, hi, with_ci=True))
        sup_lines.append(" & ".join(cells) + " \\\\")
    sup_body = "\n".join(sup_lines)

    paper_dir = os.path.join(root, "paper")
    main_tex = os.path.join(paper_dir, "main.tex")
    sup_tex = os.path.join(paper_dir, "supplementary.tex")
    # Write snippet files for manual paste, or patch files
    with open(os.path.join(out_dir, "ablation_main_table_body.tex"), "w", encoding="utf-8") as f:
        f.write(main_body)
    with open(os.path.join(out_dir, "ablation_supplementary_table_body.tex"), "w", encoding="utf-8") as f:
        f.write(sup_body)
    print("Wrote", os.path.join(out_dir, "ablation_main_table_body.tex"))
    print("Wrote", os.path.join(out_dir, "ablation_supplementary_table_body.tex"))
    print("Main table body (mean±std):")
    print(main_body)
    print("\nSupplementary table body (mean±std, 95% CI):")
    print(sup_body)

    if args.patch:
        main_tex = os.path.join(paper_dir, "main.tex")
        sup_tex = os.path.join(paper_dir, "supplementary.tex")
        with open(main_tex, "r", encoding="utf-8") as f:
            main_content = f.read()
        old_main = "Bodhi VLM (full) & 0.34 & 0.89 & 0.51 & 1.22 & \\textbf{2.02} & \\textbf{5.21} \\\\\nw/o BUA (TDA only) & 0.31 & 0.85 & 0.49 & 1.18 & 2.18 & 5.58 \\\\\nw/o TDA (BUA only) & 0.36 & 0.92 & 0.53 & 1.26 & 2.14 & 5.49 \\\\\nw/o EMPA & 0.38 & 0.95 & 0.55 & 1.31 & 3.41 & 7.02 \\\\"
        if old_main in main_content:
            main_content = main_content.replace(old_main, main_body)
            with open(main_tex, "w", encoding="utf-8") as f:
                f.write(main_content)
            print("Patched main.tex")
        else:
            print("main.tex: could not find exact table body to replace")
        with open(sup_tex, "r", encoding="utf-8") as f:
            sup_content = f.read()
        old_sup = "Component ablation (full pipeline vs.\\ w/o BUA, w/o TDA, w/o EMPA) on PPDPTS (MOT20) and COCO with Laplace noise. The main paper (Table~X) reports point estimates; full multi-seed statistics are to be populated here from runs of the component-ablation protocol over 5 random seeds. Structure: same rows as in the main paper (Bodhi VLM full, w/o BUA, w/o TDA, w/o EMPA) and same columns (Dev. and rMSE for $\\epsilon \\in \\{0.1, 0.01\\}$). Until multi-seed component-ablation runs are completed, the main table uses single-run point estimates and references this supplementary table for mean$\\pm$std and 95\\% CI.\n\n\\end{document}"
        if "\\end{document}" in sup_content:
            # Insert Table S3 before \end{document}
            tbl = (
                "\n\n\\begin{center}\n\\begin{tabular}{l|cc|cccc}\n\\toprule\n"
                "& \\multicolumn{2}{c|}{PPDPTS (MOT20)} & \\multicolumn{4}{c}{COCO (Laplace)} \\\\\n"
                "Method & Dev. $\\epsilon{=}0.1$ & Dev. $\\epsilon{=}0.01$ & Dev. $\\epsilon{=}0.1$ & Dev. $\\epsilon{=}0.01$ & rMSE $\\epsilon{=}0.1$ & rMSE $\\epsilon{=}0.01$ \\\\\n\\midrule\n"
                + sup_body +
                "\n\\bottomrule\n\\end{tabular}\n\\end{center}\n\n"
                "Source: \\texttt{ablation\\_component\\_summary.csv} from detector component-ablation (5 seeds).\n\n"
            )
            sup_content = sup_content.replace("\\end{document}", tbl + "\\end{document}")
            with open(sup_tex, "w", encoding="utf-8") as f:
                f.write(sup_content)
            print("Patched supplementary.tex")
    return main_body, sup_body

if __name__ == "__main__":
    main()
