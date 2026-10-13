"""
Fig. anytime (v2): mejor MAE de validación encontrado hasta cada evaluación
(fila superior) y hasta cada época de entrenamiento consumida (fila inferior,
la escala común que permite incluir BOHB). Media ± IC 95 % sobre semillas.

Lee outputs/analysis/anytime_by_eval.csv y anytime_by_epoch.csv (analyze.py).
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

A = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.OUT_DIR, 'analysis')
_FIG = os.path.join(C.BASE_DIR, '..', 'Figures')          # carpeta del manuscrito, si existe
OUT = _FIG if os.path.isdir(_FIG) else os.path.join(C.OUT_DIR, 'figures')
COL = {'PSO': '#0077BB', 'GA': '#EE7733', 'ABC': '#009988', 'CMAES': '#33BBEE',
       'TPE': '#CC3311', 'BOHB': '#EE3377', 'RS': '#555555'}
LBL = {'CMAES': 'CMA-ES'}

ev = pd.read_csv(os.path.join(A, 'anytime_by_eval.csv'))
ep = pd.read_csv(os.path.join(A, 'anytime_by_epoch.csv'))

plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5,
                     'axes.spines.top': False, 'axes.spines.right': False})
MK = {'PSO': 'o', 'GA': 's', 'ABC': '^', 'CMAES': 'D', 'TPE': 'v', 'BOHB': 'P', 'RS': 'X'}
fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
for j, model in enumerate(C.MODELS):
    for row, (df, x, xlabel, x0, marks) in enumerate(((ev, 'eval', 'Evaluations', 5, [10, 25, 50, 100]),
                                                      (ep, 'epoch_bin', 'Training epochs', 50, [100, 250, 500, 1000]))):
        ax = axes[row, j]
        ylo, yhi = np.inf, -np.inf
        for opt, c in COL.items():
            g = df[(df.model == model) & (df.optimizer == opt)].sort_values(x)
            if g.empty or (row == 0 and opt == 'BOHB'):
                continue                     # BOHB solo en la escala de épocas
            g = g[(g['count'] >= max(3, 0.8 * g['count'].max())) & (g[x] >= x0)]
            if g.empty:
                continue
            half = 1.96 * g['std'] / np.sqrt(g['count'])
            ax.plot(g[x], g['mean'], color=c, lw=1.2, label=LBL.get(opt, opt))
            ax.fill_between(g[x], g['mean'] - half, g['mean'] + half, color=c, alpha=0.06, lw=0)
            mk = g[g[x].isin(marks)] if row == 0 else g.iloc[[int(np.argmin(np.abs(g[x].values - v))) for v in marks]]
            ax.plot(mk[x], mk['mean'], MK[opt], color=c, ms=3.5, ls='none')
            sel = g[g[x] >= (10 if row == 0 else 100)]['mean']
            ylo, yhi = min(ylo, sel.min()), max(yhi, sel.max())
        pad = 0.08 * (yhi - ylo)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_xscale('log')
        ax.set_xticks(marks); ax.set_xticklabels(marks); ax.minorticks_off()
        ax.set_xlabel(xlabel, fontsize=8)
        ax.text(0.97, 0.95, f'({"abcdef"[row * 3 + j]})', transform=ax.transAxes,
                ha='right', va='top', fontsize=8.5, fontweight='bold')
        if row == 0:
            ax.set_title(model, fontweight='bold')
        if j == 0:
            ax.set_ylabel('Best validation MAE (W/m²)', fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=c, marker=MK[o], ms=3.5, lw=1.2) for o, c in COL.items()]
labels = [LBL.get(o, o) for o in COL]
fig.legend(handles, labels, loc='lower center', ncol=7,
           frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1))
os.makedirs(OUT, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'f6_anytime_convergence.{ext}'), dpi=300, bbox_inches='tight')
print('Guardado', os.path.abspath(os.path.join(OUT, 'f6_anytime_convergence.pdf')))
