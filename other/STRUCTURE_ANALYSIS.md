# Bodhi VLM 论文结构分析（对照 FedAFD 2603.04890）

## 一、参考论文 FedAFD 的典型结构

### 1. Related Work（分类子节）
- **2.1 Multimodal Federated Learning**：多模态联邦学习一条线
- **2.2 Feature Fusion**：特征融合一条线
- **2.3 Knowledge Distillation in Federated Learning**：联邦学习中知识蒸馏一条线

### 2. Method（分级阐述）
- **3.1 Preliminary**：问题形式化、符号、设定
- **3.2 Model Structure**：整体框架概述
  - **3.2.1 Bi-level Adversarial Alignment: BAA**
  - **3.2.2 Granularity-aware Feature Fusion: GFF**
  - **3.2.3 Similarity-guided Ensemble Distillation: SED**
- 算法用 **Algorithm** 环境（编号 Algorithm 1），不用 figure

### 3. Experiments（完整子节）
- **4.1 Experimental Setup**
  - 4.1.1 Datasets
  - 4.1.2 Evaluation Metrics
  - 4.1.3 Baselines
  - 4.1.4 Implementation Details
- **4.2 Overall Performance**
- **4.3 Ablation Study**
- **4.4 Interpretability Analysis**（如 t-SNE、特征可视化）

### 4. 理论部分
- 定理有 **正式定理陈述**，重要结论在 **Appendix** 中给出证明（如 BAA 的 domain adaptation 理论）

---

## 二、您当前论文的差距与建议

### （一）Related Work

**现状**：用 `\paragraph{}` 写了四块——差分隐私与审计、层次/多模态表示、VLM 隐私、EM 与聚类。内容有，但**没有按子节分类**，层次不如 FedAFD 清晰。

**建议**：改为带编号的子节，并按主题分类，例如：

- **2.1 Differential Privacy and Privacy Auditing**  
  保留：DP、审计、Hitaj、Abadi、McMahan 等。
- **2.2 Hierarchical and Multimodal Representations**  
  保留：FPN、CLIP、LLaVA、BLIP、Qwen2-VL、InternVL 等。
- **2.3 Privacy in Vision and Vision-Language Models**  
  保留：ViP、DP-Cap、DP-MTV、VisShield、membership inference 等。
- **2.4 Expectation-Maximization and Microaggregation**  
  保留：EM、MDAV、NCP、k-anonymity 等。

这样审稿人一眼能看出“审计”“表示”“VLM 隐私”“EM/聚类”四条线，与 FedAFD 的 Related Work 风格一致。

---

### （二）Method 结构

**现状**：
- 有 Problem Setting，但没有单独的 **Preliminary** 小节（符号、假设集中放在一起）。
- 有 BUA、TDA、EMPA、Extension to VLM，但缺少一个 **Model Structure** 总览小节（类似 FedAFD 的 3.2），在讲各模块前先画“整体图”。
- BUA、TDA 的伪代码放在 **figure** 里（`\begin{figure}...\begin{algorithmic}`），您已明确希望改为 **algorithm** 环境。
- **Theorem 1 / Theorem 2** 是手写加粗，没有用 `\begin{theorem}...\end{theorem}`，也**没有 Proof**；参考论文会把证明放在正文或 Appendix。

**建议**：

1. **增加 3.1 Preliminary**  
   - 集中给出：$D$, $S_\epsilon$, $\epsilon$, $G_i$, $G_i'$, 层索引 $i$ 等符号，以及“敏感/非敏感”“层次特征”等假设。  
   - 现有的 Problem Setting 可并入 Preliminary，或保留为 3.2 Problem Formulation。

2. **增加 3.2 Model Structure（或 Framework Overview）**  
   - 一段话 + 一张总图：BUA/TDA 如何产生 $G_i,G_i'$，EMPA 如何用它们做 budget assessment，以及如何接到 VLM。  
   - 然后再分：  
     - 3.3 Bottom-Up Strategy (BUA)  
     - 3.4 Top-Down Strategy (TDA)  
     - 3.5 Expectation-Maximization Privacy Assessment (EMPA)  
     - 3.6 Extension to Vision-Language Models  

3. **BUA / TDA 使用 algorithm 环境**  
   - 用 `\begin{algorithm}[t] ... \end{algorithm}`，内嵌 `\begin{algorithmic}...\end{algorithmic}`，带 `\caption` 和 `\label`，正文引用为 Algorithm 1/2，不再用 Figure。

4. **Theorem 规范化并补证明**  
   - 使用 `\begin{theorem}...\end{theorem}`（需 `\usepackage{amsthm}`）。  
   - **Theorem 1（E-step weight）**：给出完整陈述；证明可由 EM 的 E-step 条件期望 + 您定义的 $\omega_i(v;\hat{\Lambda})$ 推导，写进 `\begin{proof}...\end{proof}` 或 Appendix。  
   - **Theorem 2（M-step update）**：同样正式定理 + 证明（最大化 $M(\Lambda,\Lambda^l)$ 对 $\lambda_i$ 求导得闭式解）。

---

### （三）Experiments 结构

**现状**：
- 有“Experimental Protocol and Reproducibility”“Datasets and Setups”，以及 BUA vs TDA、EMPA、Comparison、Script 结果、VLM 对比等，但：
  - 没有统一的 **4.1 Experimental Setup** 父节，其下再分 Datasets / Evaluation Metrics / Baselines / Implementation Details。
  - 没有明确的 **4.2 Overall Performance** 子节（总表、总结论）。
  - Ablation 只有一段话 + “see supplementary”，没有像 FedAFD 那样的 **完整 Ablation 表格**（如 w/o BUA, w/o TDA, w/o EMPA）。
  - **缺少 Interpretability Analysis**（如 t-SNE、敏感/非敏感特征分布、层间偏差可视化等）。

**建议**：

1. **4.1 Experimental Setup**  
   - **4.1.1 Datasets**：MOT20、COCO、合成数据、VLM 用的数据及规模。  
   - **4.1.2 Evaluation Metrics**：deviation、rMSE、bias、PSNR 等定义与用途。  
   - **4.1.3 Baselines**：Chi-square、K-L、MMD、以及 ViP/DP-Cap 等对比方法。  
   - **4.1.4 Implementation Details**：网络、优化器、超参、隐私预算取值、代码/脚本说明。

2. **4.2 Overall Performance**  
   - 用 1–2 段总结：主表（如 rMSE 表、VLM 对比表）的主要结论；BUA vs TDA、EMPA vs 其他指标的整体趋势。

3. **4.3 Ablation Study**  
   - 单独小节，表格：完整 Bodhi VLM vs w/o BUA vs w/o TDA vs w/o EMPA（或只去掉其中一个策略），报告 deviation / rMSE / bias。  
   - 若部分实验在 supplementary，至少在主文有一个汇总表 + 一句“详见附录”。

4. **4.4 Interpretability Analysis**  
   - 新增小节：  
     - 例如：敏感 vs 非敏感特征在某一层的分布（直方图或 2D 投影）；  
     - BUA 与 TDA 在各层的 deviation 曲线对比；  
     - 可选：t-SNE/UMAP 可视化原始 vs 加噪后的特征，说明 EMPA 与 budget 的关系。  
   - 与 FedAFD 的 “Interpretability Analysis” 对应，突出“可解释性”。

---

## 三、您特别提到的三点——对应修改

| 您的要求 | 当前状态 | 建议 |
|----------|----------|------|
| 相关工作需要**分类介绍** | 目前是 4 个 \paragraph | 改为 2.1–2.4 子节，按 DP/表示/VLM 隐私/EM 四条线组织 |
| 提出方法要**分开阐述** | 已有 BUA/TDA/EMPA 子节，缺总览与 Preliminary | 加 Preliminary + Model Structure，再 3.3–3.6 分模块 |
| **TDA、BUA 用 algorithm 不用 figure** | 伪代码在 figure 里 | 改为 `\begin{algorithm}...\end{algorithm}`，删除 figure 包裹 |
| **Theorem 需要证明** | 只有陈述，无 proof | 用 amsthm 的 theorem + proof 环境，写 E-step/M-step 的推导 |

---

## 四、小结：建议补全与修改清单

1. **Related Work**：改为 2.1–2.4 子节，分类为 DP 与审计、层次与多模态表示、VLM 隐私、EM 与微聚合。  
2. **Method**：增加 3.1 Preliminary、3.2 Model Structure；BUA/TDA 用 algorithm 环境；Theorem 1/2 用定理环境并加证明。  
3. **Experiments**：增加 4.1 Experimental Setup（含 Datasets, Metrics, Baselines, Implementation Details）、4.2 Overall Performance、4.3 Ablation Study（含表格）、4.4 Interpretability Analysis。

按上述调整后，整体结构可与 FedAFD 的典型论文结构对齐，同时保留您工作的技术内容与贡献。
