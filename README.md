# Bodhi VLM

Privacy Budget Assessment for Vision and Vision-Language Models via Bottom-Up, Top-Down Feature Search and Expectation-Maximization Analysis.

## Structure

- **bua/** — Bodhi VLM paper (main.tex, ref.bib), images, experiment scripts (git submodule)

> 若 `bua/` 为空，执行 `git submodule update --init --recursive`。首次设置见 `SETUP_SUBMODULE.md`。

## Setup

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/mabo1215/bodhi-vlm.git

# Or if already cloned, init submodule
git submodule update --init --recursive
```

## Paper

See `bua/main.tex` for the IEEE Trans format paper. Build with:
```bash
cd bua
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
