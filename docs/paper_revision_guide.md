# Bodhi VLM 修订指南（基于 update_revision_suggestion.tex）

本文档提炼审稿意见中的具体修改要求，便于在 `paper/main.tex` 中逐项落实。若 paper 为子模块，请在拉取/检出 paper 后按下列位置修改。修订进度与每条“已改/未改”原因见 **docs/major_revision.txt**。

---

## 1. 标题与定位（Item 1）

- **修改**：标题收窄为“特征级预算对齐审计”，避免“隐私预算评估”的宽泛表述。
- **建议标题**：*Bodhi: Feature-Level Budget-Alignment Auditing for Vision Backbones and VLM Vision Encoders*
- **正文**：将“vision and vision-language models”统一改为“vision backbones and VLM vision encoders”；删除“formal privacy certification / effective ε estimation”等除非有证明的表述。

---

## 2. 摘要（Item 2）

- **修改**：明确为“经验性特征级审计框架”；说明审计在“审稿人指定的参考扰动模型”下进行。
- **必须声明不提供**：形式化差分隐私认证、端到端 VLM 隐私保证、通用有效隐私预算估计量。
- **替换表述**：将“assesses privacy budgets”改为“audits alignment between declared perturbation settings and observed feature-level perturbation patterns”。
- **可选形式化**：在摘要或方法中引入  
  \(\mathcal{A}_{\mathrm{audit}} = \mathcal{A}(\tilde{f}, f, S, \epsilon; \mathcal{M}_{\mathrm{ref}}, \Pi, \phi)\)，并说明 \(f,\tilde{f},S,\epsilon,\mathcal{M}_{\mathrm{ref}},\Pi,\phi\) 的含义。

---

## 3. 引言：问题定义（Item 3）

- **位置**：引言末尾增加一段**形式化问题定义**。
- **建议内容**：
  - **Input:** \((f, \tilde{f}, S, \epsilon, \mathcal{M}_{\mathrm{ref}})\)
  - **Output:** \(\mathrm{BAS}_\epsilon\)（预算对齐分数，衡量观测扰动特征分布与参考扰动分布之间的差异）
  - 明确写出：\(\mathrm{BAS}_\epsilon \neq \epsilon\)，且 \(\mathrm{BAS}_\epsilon\) 不蕴含 \((\epsilon,\delta)\)-DP 认证。

---

## 4. 引言：范围与局限（Item 4）

- **增加一段“范围与局限”**：
  - 对多模态模型，当前仅评估**视觉编码器/vision tower**，不评估完整语言生成链路。
  - 方法审计的是特征扰动对齐，而非生成文本的语义泄露。
  - 端到端多模态隐私为未来工作。

---

## 5. 相关工作（Items 5–6）

- **按任务重组**，分为四类：
  1. 视觉与多模态学习的形式化隐私机制；
  2. 经验性隐私审计/泄露估计方法；
  3. 敏感区域定位与特征归因方法；
  4. 分布差异估计与两样本比较方法。
- **增加定位句**：*Bodhi is not a training-time privacy mechanism; it is a post hoc feature-level auditing framework.*
- **补充基线族**：直接扰动尺度估计、似然机制拟合、分类器两样本检验、Wasserstein/energy distance、基于矩或统计量的回归预算预测等。

---

## 6. 记号与问题形式（Items 7–8）

- **敏感集**：将 \(S_\epsilon\) 改为 \(S_{\mathrm{sens}}\) 或 \(S^{(\mathrm{sens})}\)，除非有证明敏感集由预算构造。
- **审计对象与参考模型**：增加定义块，例如  
  \(f \in \mathbb{R}^d\)，\(\tilde{f} = T_{\mathrm{obs}}(f)\)，\(\tilde{f}^{\mathrm{ref}}_\epsilon = T_{\mathrm{ref},\epsilon}(f)\)，  
  并定义审计目标为  
  \(D_\epsilon = D(\mathbb{P}(\tilde{f}|f,S_{\mathrm{sens}}), \mathbb{P}(\tilde{f}^{\mathrm{ref}}_\epsilon|f,S_{\mathrm{sens}}))\) 或其由 BUA/TDA 与 EMPA 诱导的近似。

---

## 7. BUA/TDA（Items 9–11）

- **新增小节**：“Why BUA/TDA is a reasonable approximation to sensitive feature localization”：说明所用敏感信号、为何对应隐私相关、假设与预期失败情形。
- **跨层映射**：显式写出 \(\Pi_{i\to i-1}: \mathcal{F}_i \to \mathcal{F}_{i-1}\) 为近似而非精确语义对应；增加 **Assumption A1**：\(\Pi_{i\to i-1}\) 在审计目的下保持粗粒度敏感/非敏感结构。
- **复杂度**：给出每层及总复杂度，例如 \(\mathcal{O}(\sum_{i=1}^L n_i^2)\) 或 \(\mathcal{O}(\sum_{i=1}^L n_i k_i)\)。

---

## 8. EMPA（Items 12–14）

- **完整定义**：给出 \(\Theta_\epsilon^{\mathrm{obs}}\)、\(\Theta_\epsilon^{\mathrm{ref}}\)（含 \(\lambda,\mu,\Sigma\)），并定义  
  \(\mathrm{BAS}_\epsilon = d_\lambda(\lambda^{\mathrm{obs}}_\epsilon,\lambda^{\mathrm{ref}}_\epsilon) + \alpha d_\mu(\cdot) + \beta d_\Sigma(\cdot)\)，说明 \(\alpha,\beta \geq 0\)。
- **可辨识性**：增加小节讨论在何种条件下 \(\mathrm{BAS}_{\epsilon_1} \neq \mathrm{BAS}_{\epsilon_2}\) 当 \(\epsilon_1 \neq \epsilon_2\)；何时可辨识、何时失效、分组质量的作用。
- **为何用 EM**：增加一段说明用混合模型 \(\mathbb{P}(\Delta f) = \sum_k \lambda_k \mathcal{N}(\Delta f; \mu_k, \Sigma_k)\) 刻画异质扰动结构，以及潜结构相比单方差模型多出的信息。

---

## 9. 实验设计（Items 15–20）

- **围绕三个研究问题重组**：  
  **RQ1** 可辨识性；**RQ2** 分组有效性；**RQ3** 审计分数与隐私/效用结果的相关性。
- **预算敏感性实验**：固定分组、只变扰动尺度 \(\epsilon_1 < \cdots < \epsilon_m\)，报告 \(\mathrm{BAS}_\epsilon\) 是否单调或至少可区分；可报 Spearman/Kendall 与 \(\epsilon^{-1}\) 的相关。
- **机制失配实验**：\(T_{\mathrm{obs}} \neq T_{\mathrm{ref},\epsilon}\)（如观测 Gaussian vs 参考 Laplace），检验 \(\mathrm{BAS}(T_{\mathrm{obs}},T_{\mathrm{ref}}) > \mathrm{BAS}(T_{\mathrm{obs}},T_{\mathrm{obs}})\)。
- **分组消融**：与随机划分、top-k 敏感度、k-means、层次聚类、saliency 等比较；有参考敏感区域时报告 IoU/Dice。
- **阈值敏感性**：\(\tau \in \{0.80, 0.85, 0.90, 0.95\}\) 等，展示审计输出如何变化。
- **任务相关基线**：至少 (a) 直接扰动尺度估计（如 Gaussian MLE \(\hat{\sigma}^2\)）；(b) 矩回归；(c) 两样本分类器 AUC；(d) Wasserstein/energy distance 之一。

---

## 10. 统计严谨性（Items 21–22）

- **多种子**：主要表格与图报告 mean ± std（至少 N 个种子），并注明 N；可加 95% 置信区间 \(\bar{x} \pm 1.96 s/\sqrt{N}\)。
- **显著性**：对配对比较做适当检验（如 paired t-test 或 Wilcoxon）；多方法/多数据集时考虑 Holm–Bonferroni 等校正。

---

## 11. 结果解读（Items 23–24）

- **EMPA 近似常数**：增加小节“When EMPA reflects partition structure more strongly than perturbation magnitude”，承认现象、说明是否因仅用权重偏差、检验均值/协方差项是否恢复敏感性，并说明当前分数应理解为**混合结构-差异度量**而非纯预算信号。
- **审计价值**：报告 \(\mathrm{BAS}_\epsilon\) 与 MIA-AUC、ReID 泄露、\(\Delta\mathrm{AP}\) 等代理的相关系数（至少作为经验证据）。

---

## 12. VLM 部分（Items 25–26）

- **表述**：全文将“auditing VLM privacy budgets”改为“auditing the visual encoder representations of VLMs”；增加正式范围声明：*Current study scope = vision tower only, not full autoregressive multimodal generation.*
- **表格**：将 ViP/DP-Cap 等对比表改名为“Scope and capability comparison with related privacy-aware vision/VLM methods”，不做直接 SOTA 定量比较。

---

## 13. 图表与表格（Items 27–28）

- **图**：拆分为 (1) 高层框架总览；(2) BUA/TDA 细节；(3) EMPA 拟合与差异。
- **表**：每表注明种子数、指标“越高/越低越好”、归一化方式；仅在做显著性检验后标注“最佳”。

---

## 14. 讨论与局限（Item 29）

- **结构化局限**：(a) 参考模型依赖；(b) 分组质量依赖；(c) 可辨识性限制；(d) VLM 仅限视觉编码器；(e) 无形式化隐私保证。

---

## 15. 结论（Item 30）

- **重写为精确、克制**：结论中写明“本文提出一种经验性特征级审计框架，用于评估观测特征扰动与审稿人指定的参考扰动设置之间的对齐（vision backbones and VLM vision encoders）”。
- **下一步**：可辨识性分析、与隐私泄露的形式联系、扩展到完整多模态链路、跨异质扰动族的校准。

---

*修订时请同步更新 `docs/major_revision.txt` 中对应条目的进度与说明。*
