# 添加 bua 子模块

bua 内容已复制到 `C:\source\bodhi-vlm-bua` 并完成首次提交。按下列步骤将 bua 作为子模块加入本仓库。

## 1. 在 GitHub 上创建 bua 仓库

前往 https://github.com/new 新建仓库：

- **Repository name**: `bodhi-vlm-bua`（或 `bua`）
- **Public**
- 不勾选 “Add a README” 等，保持空仓库

## 2. 推送 bua 内容

```powershell
cd C:\source\bodhi-vlm-bua

# 若使用 bodhi-vlm-bua 仓库
git remote add origin https://github.com/mabo1215/bodhi-vlm-bua.git
git branch -M main
git push -u origin main

# 若改用 bua 仓库名，则：
# git remote set-url origin https://github.com/mabo1215/bua.git
# git push -u origin main
```

## 3. 添加 bua 子模块到 bodhi-vlm

```powershell
cd C:\source\bodhi-vlm

# 使用 bodhi-vlm-bua 仓库
git submodule add https://github.com/mabo1215/bodhi-vlm-bua.git bua

# 或若使用 bua 仓库：
# git submodule add https://github.com/mabo1215/bua.git bua

git add .gitmodules bua
git commit -m "Add bua as submodule"
git push origin main
```

## 4. 以后克隆含子模块的项目

```powershell
git clone --recurse-submodules https://github.com/mabo1215/bodhi-vlm.git
```

若已克隆但未拉取子模块：

```powershell
git submodule update --init --recursive
```
