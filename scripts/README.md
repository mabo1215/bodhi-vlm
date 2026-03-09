# Bodhi VLM 实验脚本

本目录包含 Bodhi VLM 论文实验的 Python 脚本，用于复现 BUA/TDA、EMPA 与 Chi-square、K-L、MMD 的对比。

## 依赖

```bash
pip install -r requirements_experiments.txt
```

或至少安装：`numpy`, `scipy`（`pandas` 和 `matplotlib` 可选，用于保存 CSV 和绘图）。

## 运行实验

```bash
# 默认：epsilon=0.1, 0.01，结果写入 results/
python run_experiments.py

# 指定输出目录和多个隐私预算
python run_experiments.py --out_dir ../results --epsilon 0.1 0.01 0.001

# 更多样本与层数（更接近 VLM 多层特征）
python run_experiments.py --n_samples 500 --n_layers 6 --epsilon 0.1 0.01
```

## 输出

- `results/bodhi_vlm_metrics.csv`：各 epsilon 下的 chi2、kl、mmd、rmse、empa_bias_bua、empa_bias_tda
- `results/bodhi_vlm_empa_bias.png`：EMPA bias 随 epsilon 变化的曲线（若已安装 matplotlib）

## 模块说明

- **metrics.py**：Chi-square、K-L、MMD、rMSE、EMPA 偏置（简化 EM 混合权重）
- **grouping.py**：BUA/TDA 风格的分组（MDAV 式聚类 + 敏感/非敏感划分）
- **run_experiments.py**：生成合成多层特征、加噪、跑分组与指标并汇总
