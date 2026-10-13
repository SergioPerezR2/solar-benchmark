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
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharey='col')
for j, model in enumerate(C.MODELS):
    for row, (df, x, xlabel) in enumerate(((ev, 'eval', 'Evaluations'),
                                           (ep, 'epoch_bin', 'Training epochs'))):
        ax = axes[row, j]
        for opt, c in COL.items():
            g = df[(df.model == model) & (df.optimizer == opt)].sort_values(x)
            if g.empty or (row == 0 and opt == 'BOHB'):
                continue                     # BOHB solo en la escala de épocas
            g = g[g['count'] >= max(3, 0.8 * g['count'].max())]
            if g.empty:
                continue
            half = 1.96 * g['std'] / np.sqrt(g['count'])
            ax.plot(g[x], g['mean'], color=c, lw=1.1, label=f'{LBL.get(opt, opt)} (n={int(g["count"].max())})')
            ax.fill_between(g[x], g['mean'] - half, g['mean'] + half, color=c, alpha=0.12, lw=0)
        for b in C.LOW_BUDGETS if row == 0 else []:
            ax.axvline(b, color='0.8', lw=0.6, ls=':', zorder=0)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.text(0.97, 0.95, f'({"abcdef"[row * 3 + j]})', transform=ax.transAxes,
                ha='right', va='top', fontsize=8.5, fontweight='bold')
        if row == 0:
            ax.set_title(model, fontweight='bold')
        if j == 0:
            ax.set_ylabel('Best validation MAE (W/m²)', fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
handles, labels = [], []
for ax in axes[1]:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l.split(' ')[0] not in [x.split(' ')[0] for x in labels]:
            handles.append(h); labels.append(l)
fig.legend(handles, [l.split(' (')[0] for l in labels], loc='lower center', ncol=7,
           frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1))
os.makedirs(OUT, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'f6_anytime_convergence.{ext}'), dpi=300, bbox_inches='tight')
print('Guardado', os.path.abspath(os.path.join(OUT, 'f6_anytime_convergence.pdf')))
