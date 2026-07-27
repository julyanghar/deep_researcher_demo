"""Two-sided exploitation evidence (skill: nature-figure, backend: python).
FIGURE CONTRACT
  Core conclusion: the same recycling property pays on BOTH sides of inference:
    (a) decode: suffix-drafter acceptance splits by call type (61.2% report vs
        28.8% summary) -> copy structure decides drafter payoff -> routing;
    (b) prefill: blended KV reuse cuts large-prompt prefill by 51-67%
        (writer 3.044->0.998 s, supervisor 1.859->0.903 s).
  Evidence chain: panel a = decode-side evidence; panel b = prefill-side
    evidence; together they defend "exploit recycling on both sides".
  Archetype: quantitative grid (2 panels). Backend: python (exclusive).
  Export: AAAI single column (3.3 in), SVG+PDF editable text + PNG preview.
  Data (measured, 100-task run provenance; regenerate on 800-task trace):
    acceptance: e2e-3config-40q traj  (report 61.2%, summary 28.8%)
    prefill s: RUN_sync_0614 TP4 replay (writer n=19/20, supervisor n=35/33)
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})

REPORT, SUMMARY = '#D55E00', '#0072B2'   # 与图 copyrate-dist 同色 (Okabe-Ito)
NATIVE, BLEND   = '#999999', '#009E73'   # 中性 vs 收益色

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.3, 1.9), gridspec_kw={'wspace': 0.55})

# ---- panel a: decode 侧接受率分裂 ----
acc = [61.2, 28.8]
bars = ax1.bar([0, 1], acc, width=0.62, color=[REPORT, SUMMARY])
for x, v in zip([0, 1], acc):
    ax1.text(x, v + 2, f'{v:.1f}%', ha='center', fontsize=7)
ax1.set_xticks([0, 1]); ax1.set_xticklabels(['Report\ncalls', 'Summary\ncalls'])
ax1.set_ylabel('Suffix-drafter acceptance (%)')
ax1.set_ylim(0, 75)
ax1.set_title('Decode side', fontsize=7.5, pad=3)

# ---- panel b: prefill 侧 blend vs native ----
groups = ['Writer', 'Supervisor']
native = [3.044, 1.859]
blend  = [0.998, 0.903]
x = np.arange(2); w = 0.36
ax2.bar(x - w/2, native, w, color=NATIVE, label='Full prefill')
ax2.bar(x + w/2, blend,  w, color=BLEND,  label='Blended reuse')
for xi, (nv, bv) in enumerate(zip(native, blend)):
    ax2.text(xi, nv + 0.12, f'−{(1-bv/nv)*100:.0f}%', ha='center', fontsize=7, color=BLEND)
ax2.set_xticks(x); ax2.set_xticklabels(groups)
ax2.set_ylabel('Prefill time (s)')
ax2.set_ylim(0, 4.4)
ax2.legend(fontsize=6, loc='upper right', handlelength=1.2)
ax2.set_title('Prefill side', fontsize=7.5, pad=3)

for ax, lab in [(ax1, 'a'), (ax2, 'b')]:
    ax.text(-0.32, 1.08, lab, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')

fig.tight_layout()
import os
os.chdir('/home/yilin/deep_researcher_demo/exp-docx/paper-submission/AAAI/Figures')
fig.savefig('twosides.svg', bbox_inches='tight')
fig.savefig('twosides.pdf', bbox_inches='tight')
fig.savefig('twosides.png', dpi=300, bbox_inches='tight')
print("saved: twosides.{svg,pdf,png}")
