# SpecForge 本地 fork 备份(git bundle)

EAGLE-3 域训(phase2-12k-3ep 权重)用的训练框架是 SpecForge 本地 fork,
四补丁(ropebuf / chunk-acc / nocompile / ckpt-norm)+ 位置维裁剪(trim A/B)都在本地分支里。
直接把仓库嵌套进本仓库会产生裸 gitlink 坏克隆(本仓库 commit a0b0a5b 曾为此清理过论文仓库的 gitlink),
所以用 `git bundle` 打成单文件,普通文件不影响克隆。

## 内容

| 文件 | 说明 |
|---|---|
| `specforge-all.bundle` | 全部分支+全历史(`git bundle create --all`,含 main / pr-ropebuf* / pr-trim-a* / trim-usp-dev / yilin-trim-mem-probes) |
| `untracked-bak-files.tar.gz` | 9 个未跟踪的 `.bak_*` 文件(打补丁前的原件快照) |

打包时(2026-08-16)的分支头:

```
main                  357a97e  (上游 sgl-project/SpecForge, behind 336)
pr-ropebuf            66825ad
pr-ropebuf-v2         7a1040f  (= 公开 fork julyanghar/SpecForge 的 pr-ropebuf)
pr-trim-a             f31dce1
pr-trim-a-v2          ba20730
pr-trim-a-v3          efbc096  (= 公开 fork 的 pr-trim-a,PR #705,训练用的就是这条线)
trim-usp-dev          605d56c
yilin-trim-mem-probes 78a6432
```

## 恢复

```bash
git clone specforge-all.bundle SpecForge      # 得到完整仓库
cd SpecForge
git branch -a                                  # 所有分支都在
git remote set-url origin https://github.com/sgl-project/SpecForge.git
tar xzf ../untracked-bak-files.tar.gz          # 需要 .bak 原件时
```

训练环境配置见 [../env/](../env/) 无此文件时看 SpecForge 仓库内 `env/`(specforge conda env,
Python 3.12 / torch 2.11.0+cu130)。训练命令与数据谱系见
[../phase2-12k-3ep/README.md](../phase2-12k-3ep/README.md)。
