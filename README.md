# Bodhi VLM

Privacy Budget Assessment for Vision and Vision-Language Models via Bottom-Up, Top-Down Feature Search and Expectation-Maximization Analysis.

## Structure

- **main.tex** — 主文稿（IEEE Trans 格式）
- **ref.bib** — 参考文献
- **images/** — 图片
- **scripts/** — 实验脚本
- **results/** — 实验输出

## 编译

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## 实验

```bash
cd scripts
pip install -r requirements_experiments.txt
python run_experiments.py --out_dir ../results --epsilon 0.1 0.01 0.001
```
