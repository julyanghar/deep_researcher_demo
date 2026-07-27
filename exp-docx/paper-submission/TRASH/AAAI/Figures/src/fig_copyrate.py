"""Figure: per-call copy-rate distribution (bimodal), real data from per_call_metrics.csv.
skill: scientific-visualization. Data: 100-task run (online100_v2), 1,912 calls.
NOTE provenance: paper headline is the 800-task trace; regenerate on it before submission."""
import sys, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 字体: Liberation Sans(Arial 度量兼容, 已装 ~/.fonts), 免 findfont 告警
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'DejaVu Sans']

sys.path.insert(0, '/home/yilin/.claude/skills/scientific-visualization/scripts')
from figure_export import save_publication_figure
from style_presets import apply_publication_style
apply_publication_style('default')
plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'DejaVu Sans']  # style 后再钉一次

# ---- 读真实数据 ----
rows = list(csv.DictReader(open('/home/yilin/deep_researcher_demo/exp-docx/TRASH/prompt-overlap-analysis-v2/per_call_metrics.csv')))
summ = [float(r['cov8_raw'])*100 for r in rows if r['tag'] == 'RESEARCH_SUMMARY_TEXT']
rep  = [float(r['cov8_raw'])*100 for r in rows if r['tag'] == 'FINAL_REPORT_MARKDOWN']
print(f"[数据核对] summary n={len(summ)} mean={np.mean(summ):.1f}% median={np.median(summ):.1f}% | report n={len(rep)} mean={np.mean(rep):.1f}%")

# ---- 画图 (AAAI 单栏 ~3.3in) ----
OI_BLUE, OI_VERM = '#0072B2', '#D55E00'   # Okabe-Ito, 色盲安全
fig, ax = plt.subplots(figsize=(3.3, 2.3))
bins = np.arange(0, 105, 5)
ax.hist(summ, bins=bins, weights=np.ones(len(summ))/len(summ), color=OI_BLUE,
        alpha=0.75, label=f'Summary calls (n={len(summ)})', edgecolor='white', linewidth=0.4)
ax.hist(rep, bins=bins, weights=np.ones(len(rep))/len(rep), color=OI_VERM,
        alpha=0.75, label=f'Report calls (n={len(rep)})', edgecolor='white', linewidth=0.4)
ax.set_xlabel('Per-call copy rate (%)')
ax.set_ylabel('Fraction of calls')
ax.set_xlim(0, 100)
ax.legend(frameon=False, fontsize=7, loc='upper left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout()

import os
os.chdir('/home/yilin/deep_researcher_demo/exp-docx/paper-submission/AAAI/Figures')
save_publication_figure(fig, 'copyrate-dist', formats=['pdf', 'png'], dpi=300)
