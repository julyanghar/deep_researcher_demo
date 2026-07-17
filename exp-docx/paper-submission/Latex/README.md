# Latex/ 工作流说明

这个目录是论文的 LaTeX 项目，**由脚本从 `../paper-draft.md` 生成**。不要直接在这里写正文。

## 三步循环

1. 改正文 → 编辑 `../paper-draft.md`（Markdown，唯一内容源）。
2. 生成 → `cd ..  && python3 md2tex.py paper-draft.md --split Latex`
3. 编译 → 在 VS Code 里对 **main.tex** 按 BUILD（LaTeX Workshop），或命令行 `latexmk -pdf main.tex`。

## 文件归属（关键，别搞混）

| 文件 | 谁管 | 说明 |
|---|---|---|
| `main.tex` | **脚本**（每次重生成） | 根文件。含 `\documentclass` + `\input{preamble}` + `\input` 各节 + bib。**别手改**，改了下次生成就没了。 |
| `preamble.tex` | **人**（只创建一次，脚本永不覆盖） | 宏包与自定义宏。加包、改样式在这里。**不含 `\documentclass`**。 |
| `section/*.tex` | **脚本**（每次重生成） | 每节一个，body-only。**别手改**，改 `paper-draft.md`。 |
| `refs.bib` | **人**（只创建一次） | 现在是占位 `@misc`，投稿前换成真 bibtex。脚本不覆盖已存在的它。 |

## venue = AAAI-27（2026-07-16 起）

- 模板来自 `AuthorKit27.zip`（已解包到 `authorkit27/`），`aaai2027.sty` + `aaai2027.bst` 已复制到本目录。
- **必须用 pdfLaTeX 编译**：aaai2027.sty 检测引擎、XeTeX/tectonic 直接拒绝。本机装了 TinyTeX（`~/.TinyTeX`），VS Code 默认 recipe 已指向它的 latexmk；命令行：`~/.TinyTeX/bin/x86_64-linux/latexmk -pdf main.tex`。
- **禁 hyperref**（sty 会报错）及 geometry/float/multicol 等一整页清单（见 `authorkit27/AuthorKit27/AnonymousSubmission2027.tex` 头部注释）。加宏包前先查那个清单。
- **不要写 `\bibliographystyle`**：sty 已内置 aaai2027.bst。
- 引用是**作者-年份制**（natbib）：md 里 `[@key]` 生成 `\citep`（括号引用）；需要行文主语引用时在 md 里直接写 `\citet{key}`。
- camera-ready 时把 preamble.tex 里 `[submission]` 选项去掉。

## 常见操作

- **换 venue 文档类**（如 `acmart`/`mlsys`）：改 `main.tex` 第 5 行的 `\documentclass` 那一句——但因为 main.tex 会被重生成，真正要改的是 `../md2tex.py` 顶部的 `DOCUMENTCLASS` 常量（改一次，之后每次生成都对）。venue 专属宏包加进 `preamble.tex`。
- **加新章节**：在 `paper-draft.md` 里写 `## 2 Background ...`，跑脚本 → 自动生成 `section/background.tex` 并在 main.tex 补 `\input` 行。
- **根文件识别**：只有 main.tex 含 `\documentclass` 声明，且所有 `.tex` 顶部有 `% !TEX root` 注释指向 main.tex——所以在任意子文件里按 BUILD，编辑器都会正确编译 main.tex（曾因 `\documentclass` 放在 preamble 里导致误认根，已修）。

## 校验

`md2tex.py --split` 每次生成后自动跑校验：`\input` 目标存在、`\cite` key 都在 bib、各文件括号/`$`配平、正文无漏转义的 `%` 或残留 Unicode。看到 `✓ 结构/引用/括号/转义 全部通过` 即结构无编译阻塞。

## 数字来源提醒

`paper-draft.md` 顶部有 NUMBER PROVENANCE FLAG：abstract/intro 的标题数是 800 题 / 15,296 调用，但正文细节数（照抄率、加速比等）目前仍来自最初 100 题 run，投稿前需在 800 题 run 上重算。
