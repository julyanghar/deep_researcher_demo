#!/usr/bin/env python3
"""md2tex.py — 本项目论文 md 草稿 → LaTeX。

两种输出模式：
  单文件:  python3 md2tex.py paper-draft.md [out.tex]
  多文件:  python3 md2tex.py paper-draft.md --split Latex
           生成:  Latex/preamble.tex   (手写归属，只在缺失时创建，之后永不覆盖)
                  Latex/main.tex        (脚本归属，每次重新生成：标题+abstract+\\input 各节)
                  Latex/section/<slug>.tex  (脚本归属，每节一个，body-only)
                  Latex/refs.bib        (只在缺失时创建：从 md 里的 CITATION KEY MAP 生成占位条目)

md 约定：
  # 一级标题            -> \\title{...}
  ## Abstract           -> section/abstract.tex + main.tex 里 \\begin{abstract}\\input{...}\\end{abstract}
  ## N Section Name     -> \\section{Section Name} (脱去开头编号，交给 LaTeX 自动编号)
  ### N.M Sub Name      -> \\subsection{Sub Name}
  **粗体** / *斜体*      -> \\textbf{} / \\emph{}
  [@key] / [@a; @b]     -> \\cite{key} / \\cite{a,b}
  [TBD] / [TBD-xxx]     -> \\tbd / \\textbf{[TBD-xxx]}
  1. / - 列表           -> enumerate / itemize
  <!-- ... -->          -> 单文件模式转 % 注释；--split 模式整块剥离（不进正文），仅解析 KEY MAP
  特殊字符   % & # _     -> \\% \\& \\# \\_
  Unicode  — – × ≥ ≤ → ≈ − ~N  Δ · M·ΔV
  直引号 "..."           -> ``...''
"""
import os
import re
import sys

# \documentclass 必须留在 main.tex（唯一含它的文件），否则编辑器(LaTeX Workshop)会把
# preamble.tex 误判成根文件。AAAI-27 要求 letterpaper article，禁止改动。
DOCUMENTCLASS = r"\documentclass[letterpaper]{article} % AAAI-27 AuthorKit: DO NOT CHANGE"

# preamble.tex 的内容：只放宏包与宏，不含 \documentclass。手写归属，脚本只创建一次。
# 逐行来自 AuthorKit27/AnonymousSubmission2027.tex 的必需头（标 DO NOT CHANGE 的部分）。
# ⚠ AAAI 禁 hyperref（aaai2027.sty 会直接 PackageError），也禁 geometry/float/multicol 等，别加。
PREAMBLE = r"""% !TEX root = main.tex
% preamble.tex --- hand-owned. AAAI-27 required header; aaai2027.sty forbids hyperref etc.
\usepackage[submission]{aaai2027}  % DO NOT CHANGE (camera-ready 时去掉 [submission])
\usepackage[hyphens]{url}  % DO NOT CHANGE
\usepackage{graphicx} % DO NOT CHANGE
\urlstyle{rm} % DO NOT CHANGE
\def\UrlFont{\rm}  % DO NOT CHANGE
\usepackage{natbib}  % DO NOT CHANGE, no options
\usepackage{caption} % DO NOT CHANGE, no options
\frenchspacing  % DO NOT CHANGE
\usepackage{booktabs}
\usepackage{amsmath, amssymb}
\ifdefined\pdfinfo\pdfinfo{/TemplateVersion (2027.1)}\fi % pdfTeX 专有原语；tectonic(XeTeX) 下自动跳过
\setcounter{secnumdepth}{0} % AAAI 默认小节不编号；要编号可改 1/2
\newcommand{\tbd}{\textbf{[TBD]}} % pre-registered experiment, number not yet obtained
"""


def convert_inline(s: str) -> str:
    """段内文本的全部行内转换。顺序敏感：先备份引用/TBD/宏，最后转义特殊字符。"""
    cites = []
    def _cite(m):
        keys = ",".join(k.strip().lstrip("@") for k in m.group(1).split(";"))
        cites.append(keys)
        return f"\x00CITE{len(cites)-1}\x00"
    s = re.sub(r"\s*\[(@[^\]]+)\]", _cite, s)  # 吞引用前空格，统一由 ~ 供不断行空格

    s = s.replace("[TBD]", "\x00TBD\x00")
    s = re.sub(r"\[TBD-([A-Za-z0-9-]+)\]", lambda m: f"\x00TBDX{m.group(1)}\x00", s)

    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)             # 粗体先
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\emph{\1}", s)   # 再斜体

    s = re.sub(r"\br = (−|-)?(\d+\.\d+)", lambda m: f"$r = {'-' if m.group(1) else ''}{m.group(2)}$", s)

    # 特殊字符转义（先把已生成的 \textbf{ / \emph{ 用哨兵护住）
    s = s.replace("\\textbf{", "\x00BF\x00").replace("\\emph{", "\x00EM\x00")
    for ch, rep in [("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(ch, rep)
    s = s.replace("\x00BF\x00", "\\textbf{").replace("\x00EM\x00", "\\emph{")

    s = s.replace("M·ΔV", r"$M{\cdot}\Delta V$")  # 专名整体转，避免拆成三段数学
    s = (s.replace("—", "---").replace("–", "--")
          .replace("×", r"$\times$").replace("≥", r"$\geq$").replace("≤", r"$\leq$")
          .replace("→", r"$\rightarrow$").replace("≈", r"$\approx$").replace("−", "-")
          .replace("Δ", r"$\Delta$").replace("·", r"$\cdot$"))
    s = re.sub(r"~(?=\d)", r"${\\sim}$", s)  # ~19 -> ${\sim}$19

    if s.count('"') % 2 == 0:
        s = re.sub(r'"([^"]*)"', r"``\1''", s)
    elif '"' in s:
        print(f"  [warn] 奇数个双引号，未转换: {s[:60]}...", file=sys.stderr)

    s = s.replace("vs. ", "vs.\\ ")

    s = s.replace("\x00TBD\x00", r"\tbd")
    s = re.sub("\x00TBDX([A-Za-z0-9-]+)\x00", r"\\textbf{[TBD-\1]}", s)
    # AAAI 用 natbib 作者-年份制：默认括号引用 \citep（正文主语引用需 \citet 时手写进 md）
    s = re.sub("\x00CITE(\\d+)\x00", lambda m: f"~\\citep{{{cites[int(m.group(1))]}}}", s)
    return s


def emit_body(lines):
    """一节的 md 行 -> tex 正文（处理 ### 子标题 / 列表 / 段落；不含 ## 与标题）。"""
    out, in_enum, in_item = [], False, False

    def close_lists():
        nonlocal in_enum, in_item
        if in_enum:
            out.append(r"\end{enumerate}"); in_enum = False
        if in_item:
            out.append(r"\end{itemize}"); in_item = False

    for line in lines:
        m = re.match(r"^### (.*)", line)
        if m:
            close_lists()
            h = re.sub(r"^\d+(\.\d+)* ", "", m.group(1).strip())
            out.append(rf"\subsection{{{convert_inline(h)}}}")
            continue
        m = re.match(r"^\s*\d+\.\s+(.*)", line)
        if m:
            if not in_enum:
                close_lists(); out.append(r"\begin{enumerate}"); in_enum = True
            out.append("  \\item " + convert_inline(m.group(1))); continue
        m = re.match(r"^\s*[-*]\s+(.*)", line)
        if m:
            if not in_item:
                close_lists(); out.append(r"\begin{itemize}"); in_item = True
            out.append("  \\item " + convert_inline(m.group(1))); continue
        if line.strip() == "":
            close_lists(); out.append("")
        else:
            close_lists(); out.append(convert_inline(line))
    close_lists()
    body = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"


def parse_keymap(md: str):
    """从 CITATION KEY MAP 注释块解析 {key: description}。"""
    m = re.search(r"CITATION KEY MAP.*?:\s*\n(.*?)(?:===|-->|\Z)", md, re.DOTALL)
    if not m:
        return {}
    keymap = {}
    for line in m.group(1).splitlines():
        km = re.match(r"^\s*@?([\w-]+)\s{2,}(.+?)\s*$", line)
        if km:
            keymap[km.group(1)] = km.group(2)
    return keymap


def parse_doc(md: str):
    """-> (title, [(kind, name, lines)])，kind in {'abstract','section'}。已剥离 HTML 注释。"""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)  # split 模式：注释不进正文
    title, sections = None, []
    cur_name, cur_kind, cur_lines = None, None, []

    def flush():
        if cur_kind:
            sections.append((cur_kind, cur_name, cur_lines))

    for line in md.split("\n"):
        m = re.match(r"^# (?!#)(.*)", line)
        if m:
            title = convert_inline(m.group(1).strip()); continue
        m = re.match(r"^## (.*)", line)
        if m:
            flush()
            h = m.group(1).strip()
            if h.lower() == "abstract":
                cur_kind, cur_name = "abstract", "Abstract"
            else:
                cur_kind, cur_name = "section", re.sub(r"^\d+ ", "", h)
            cur_lines = []
            continue
        if cur_kind:
            cur_lines.append(line)
    flush()
    return title, sections


def slugify(name: str) -> str:
    name = name.split(":")[0]
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "section"


def bib_entry(key: str, desc: str) -> str:
    ax = re.search(r"arXiv:(\d{4}\.\d{4,5})", desc)
    yr = re.search(r"\b(20\d\d)\b", desc)
    # AAAI(natbib 作者-年份制)：没有 author 字段引用标签会渲染成空/乱码，
    # 占位条目用 key 当"机构作者"（双花括号防拆分），正文里显式可见便于替换。
    fields = [f"  author       = {{{{{key}}}}}",
              f"  title        = {{{desc}}}"]
    if ax:
        fields.append(f"  howpublished = {{arXiv:{ax.group(1)}}}")
    if not yr and ax:  # 无显式年份时从 arXiv 号推导（2601.x → 2026），避免作者-年份制渲染成 ????
        fields.append(f"  year         = {{20{ax.group(1)[:2]}}}")
    elif yr:
        fields.append(f"  year         = {{{yr.group(1)}}}")
    fields.append("  note         = {PLACEHOLDER generated by md2tex.py -- replace with real bibtex}")
    return "@misc{" + key + ",\n" + ",\n".join(fields) + "\n}\n"


def write_split(md: str, outdir: str):
    title, sections = parse_doc(md)
    keymap = parse_keymap(md)
    secdir = os.path.join(outdir, "section")
    os.makedirs(secdir, exist_ok=True)

    # 1) preamble.tex — 只在缺失时创建
    pre = os.path.join(outdir, "preamble.tex")
    if not os.path.exists(pre):
        open(pre, "w", encoding="utf-8").write(PREAMBLE)
        print(f"created (hand-owned, will not overwrite): {pre}")
    else:
        print(f"kept (hand-owned): {pre}")

    # 2) 各节 body-only 文件（带 magic root 注释，指向上一级 main.tex，供编辑器定位根）
    ordered, abstract_slug = [], None
    for kind, name, lines in sections:
        slug = "abstract" if kind == "abstract" else slugify(name)
        path = os.path.join(secdir, f"{slug}.tex")
        sec_header = "" if kind == "abstract" else f"\\section{{{name}}}\n\n"
        open(path, "w", encoding="utf-8").write("% !TEX root = ../main.tex\n" + sec_header + emit_body(lines))
        print(f"written: {path}")
        if kind == "abstract":
            abstract_slug = slug
        else:
            ordered.append(slug)

    # 3) main.tex — 每次重生成。含 \documentclass（唯一），故编辑器把它认作根。
    L = ["% !TEX root = main.tex",
         "% main.tex --- GENERATED by md2tex.py from paper-draft.md. Do not hand-edit.",
         "% Regenerate:  python3 ../md2tex.py ../paper-draft.md --split .",
         "% Packages/macros live in preamble.tex (hand-owned); venue class = the line below.",
         DOCUMENTCLASS,
         "\\input{preamble}", ""]
    if title:
        # AAAI-27 作者块：submission 模式匿名；\affiliations 是 aaai2027.sty 提供的命令
        L += [f"\\title{{{title}}}", "\\author{Anonymous submission}", "\\affiliations{Anonymous submission}", ""]
    L += ["\\begin{document}", "\\maketitle", ""]
    if abstract_slug:
        L += ["\\begin{abstract}", "\\input{section/abstract}", "\\end{abstract}", ""]
    for slug in ordered:
        L.append(f"\\input{{section/{slug}}}")
    # AAAI: 不写 \bibliographystyle —— aaai2027.sty 已内置设置为 aaai2027.bst
    L += ["", "\\bibliography{refs}", "", "\\end{document}", ""]
    open(os.path.join(outdir, "main.tex"), "w", encoding="utf-8").write("\n".join(L))
    print(f"written: {os.path.join(outdir, 'main.tex')}")

    # 4) refs.bib — 只在缺失时创建
    bib = os.path.join(outdir, "refs.bib")
    if not os.path.exists(bib) and keymap:
        with open(bib, "w", encoding="utf-8") as f:
            f.write("% GENERATED placeholder bib by md2tex.py. Replace entries with real bibtex.\n\n")
            for k, d in keymap.items():
                f.write(bib_entry(k, d) + "\n")
        print(f"created ({len(keymap)} placeholder entries): {bib}")
    elif os.path.exists(bib):
        print(f"kept (hand-owned): {bib}")

    validate_split(outdir, keymap)


def validate_split(outdir, keymap):
    """确定性校验：\\input 目标存在 / \\cite key 都在 bib / 各文件括号配平 / 正文无裸 %。"""
    problems = []
    main = open(os.path.join(outdir, "main.tex"), encoding="utf-8").read()
    for tgt in re.findall(r"\\input\{([^}]+)\}", main):
        p = os.path.join(outdir, tgt if tgt.endswith(".tex") else tgt + ".tex")
        if not os.path.exists(p):
            problems.append(f"main.tex 的 \\input{{{tgt}}} 指向不存在的文件")

    bib_keys = set()
    bibp = os.path.join(outdir, "refs.bib")
    if os.path.exists(bibp):
        bib_keys = set(re.findall(r"@\w+\{([\w-]+),", open(bibp, encoding="utf-8").read()))
    used = set()
    for root, dirs, files in os.walk(outdir):
        dirs[:] = [d for d in dirs if d != "authorkit27"]  # kit 自带示例不归我们 lint
        for fn in files:
            if not fn.endswith(".tex"):
                continue
            path = os.path.join(root, fn)
            tex = open(path, encoding="utf-8").read()
            for m in re.findall(r"\\cite\{([^}]+)\}", tex):
                used.update(k.strip() for k in m.split(","))
            problems += lint(tex, os.path.relpath(path, outdir))
    missing = sorted(used - bib_keys - set(keymap))
    if missing:
        problems.append(f"\\cite 用到但 bib 里没有的 key: {missing}")

    print("=== 校验 ===")
    print("\n".join("  ✗ " + p for p in problems) if problems else "  ✓ 结构/引用/括号/转义 全部通过")
    return problems


def lint(tex: str, name: str):
    """正文行里未转义 % / 残留 Unicode / 括号配平。"""
    problems = []
    for n, line in enumerate(tex.split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue
        # 只抓"字面百分号漏转义"（前面紧挨非空白、且非反斜杠，如 90%）；
        # 行尾注释 " % ..."（%前是空格）与已转义 \% 都放行。
        for m in re.finditer(r"(?<=\S)(?<!\\)%", line):
            problems.append(f"{name}:{n}: 未转义的 % ...{line[max(0,m.start()-16):m.start()+8]}...")
        for ch in "—–×≥≤→≈−“”‘’":
            if ch in line:
                problems.append(f"{name}:{n}: 残留 Unicode {ch!r}")
    if tex.count("{") != tex.count("}"):
        problems.append(f"{name}: 花括号不配平 {{={tex.count('{')} }}={tex.count('}')}")
    return problems


def convert_single(md: str) -> str:
    """单文件：title + abstract 环境 + 各节，拼成一个可编译 tex。"""
    title, sections = parse_doc(md)
    L = [DOCUMENTCLASS, PREAMBLE.rstrip()]
    if title:
        L += [f"\\title{{{title}}}", "\\author{Anonymous submission}", "\\affiliations{Anonymous submission}"]
    L += ["", "\\begin{document}", "\\maketitle", ""]
    for kind, name, lines in sections:
        if kind == "abstract":
            L += ["\\begin{abstract}", emit_body(lines).rstrip(), "\\end{abstract}", ""]
        else:
            L += [f"\\section{{{name}}}", "", emit_body(lines).rstrip(), ""]
    L += ["\\bibliography{refs}", "", "\\end{document}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--split" in argv:
        i = argv.index("--split")
        src, outdir = argv[0], argv[i + 1]
        write_split(open(src, encoding="utf-8").read(), outdir)
    else:
        args = [a for a in argv if not a.startswith("--")]
        src = args[0]
        dst = args[1] if len(args) > 1 else re.sub(r"\.md$", "", src) + ".tex.gen"
        tex = convert_single(open(src, encoding="utf-8").read())
        open(dst, "w", encoding="utf-8").write(tex)
        print(f"written: {dst}")
        for p in lint(tex, dst):
            print("  LINT:", p)
