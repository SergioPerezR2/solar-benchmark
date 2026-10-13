"""
Figuras de resultados v2 (todas con paneles rotulados (a), (b), ...).

    python figures/results_figures.py [carpeta_de_analisis]

Lee los CSV de analyze.py (por defecto outputs/analysis) y escribe en ../Figures:
  f1_test_mae_distribution   distribución del MAE de prueba por optimizador
  f2_reference_comparison    modelos ajustados vs. por defecto y referencias
  f3_low_budget              MAE de prueba del incumbente a b = 10, 25, 50, 100
  f4_scatter_val_vs_test     MAE de validación vs. prueba
  f5_hp_distributions        hiperparámetros ganadores (B = 100)
  f7_hp_importance           importancia de hiperparámetros (evaluaciones de RS)
  f8_equivalence             diferencias pareadas con IC 90 % frente a ±δ
  f9_single_run              qué concluiría un estudio de una sola corrida
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
OPTS = ['PSO', 'GA', 'ABC', 'CMAES', 'TPE', 'BOHB', 'RS']
COL = {'PSO': '#0077BB', 'GA': '#EE7733', 'ABC': '#009988', 'CMAES': '#33BBEE',
       'TPE': '#CC3311', 'BOHB': '#EE3377', 'RS': '#555555', 'DEFAULT': '#BBBBBB'}
LBL = {'CMAES': 'CMA-ES', 'DEFAULT': 'Default'}
lab = lambda o: LBL.get(o, o)
PANEL = 'abcdefghij'
MODELS = C.MODELS

plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False})
runs = pd.read_csv(os.path.join(A, 'runs_long.csv'))
core = runs[runs.tag == 'core']
b100 = core[core.budget == 'B100']
refs = pd.read_csv(os.path.join(A, 'reference_forecasts.csv'))
smart = refs.query("reference=='smart_persistence' and split=='test' and block=='all'").mae.iloc[0]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Guardado', name)


def title(ax, i, text):
    ax.set_title(f'({PANEL[i]}) {text}', fontsize=9, fontweight='bold')


# ── f1: distribución del MAE de prueba ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
for i, m in enumerate(MODELS):
    ax = axes[i]
    data = [b100[(b100.model == m) & (b100.optimizer == o)].all_mae.values for o in OPTS]
    vp = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, o in zip(vp['bodies'], OPTS):
        body.set_facecolor(COL[o]); body.set_alpha(0.35); body.set_edgecolor('none')
    for k, d in enumerate(data, 1):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.vlines(k, q1, q3, color='k', lw=2.2)
        ax.scatter(k, med, color='white', s=10, zorder=3, edgecolor='k', lw=0.5)
    ax.axhline(smart, color='k', ls='--', lw=0.8)
    if i == 0:
        ax.text(0.6, smart + 0.12, 'Smart persistence', fontsize=6.5, va='bottom', ha='left')
    ax.set_xticks(range(1, 8)); ax.set_xticklabels([lab(o) for o in OPTS], rotation=45, ha='right', fontsize=7.5)
    title(ax, i, m)
    ax.grid(axis='y', alpha=0.25, lw=0.5)
axes[0].set_ylabel('Test MAE, all hours (W/m²)')
save(fig, 'f1_test_mae_distribution')

# ── f2: ajustados vs. por defecto vs. referencias ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for i, (col, blk, name) in enumerate((('all_mae', 'all', 'All hours'),
                                      ('day_ghi50_mae', 'day_ghi50', 'Daytime (GHI > 50 W/m²)'))):
    ax = axes[i]
    x = np.arange(len(MODELS)); w = 0.1
    for k, o in enumerate(OPTS + ['DEFAULT']):
        g = b100[b100.optimizer == o].groupby('model')[col].agg(['mean', 'std']).reindex(MODELS)
        ax.bar(x + (k - 3.5) * w, g['mean'], w, yerr=g['std'], color=COL[o], label=lab(o),
               error_kw={'lw': 0.6, 'capsize': 1.2})
    for ref, ls in (('persistence', ':'), ('smart_persistence', '--')):
        v = refs.query(f"reference=='{ref}' and split=='test' and block=='{blk}'").mae.iloc[0]
        ax.axhline(v, color='k', ls=ls, lw=0.9, label=ref.replace('_', ' ').capitalize())
    ax.set_xticks(x); ax.set_xticklabels(MODELS)
    ax.set_ylabel('Test MAE (W/m²)')
    title(ax, i, name)
    ax.grid(axis='y', alpha=0.25, lw=0.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.13))
save(fig, 'f2_reference_comparison')

# ── f3: presupuestos bajos ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
bud = [10, 25, 50, 100]
for i, m in enumerate(MODELS):
    ax = axes[i]
    for o in OPTS:
        g = core[(core.model == m) & (core.optimizer == o)]
        st = g.groupby('budget').all_mae.agg(['mean', 'std', 'count'])
        st = st.reindex([f'B{b}' for b in bud])
        half = 1.96 * st['std'] / np.sqrt(st['count'])
        off = (OPTS.index(o) - 3) * 0.012
        xs = np.array(bud) * (1 + off)
        ax.errorbar(xs, st['mean'], yerr=half, color=COL[o], lw=1, marker='o', ms=2.5,
                    capsize=1.5, label=lab(o))
    ax.set_xscale('log'); ax.set_xticks(bud); ax.set_xticklabels(bud)
    ax.minorticks_off()
    title(ax, i, m)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].set_ylabel('Test MAE (W/m²)')
fig.supxlabel('Evaluations used to select the incumbent ($b$)', fontsize=8.5, y=-0.05)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=7, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.17))
save(fig, 'f3_low_budget')

# ── f4: validación vs. prueba ─────────────────────────────────────────────────
vt = pd.read_csv(os.path.join(A, 'val_test_correlation.csv')).set_index('model')
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for i, m in enumerate(MODELS):
    ax = axes[i]
    g = b100[(b100.model == m) & (b100.optimizer != 'DEFAULT')]
    for o in OPTS:
        h = g[g.optimizer == o]
        ax.scatter(h.best_val_mae, h.all_mae, s=7, color=COL[o], alpha=0.7, lw=0, label=lab(o))
    ax.set_xlabel('Best validation MAE (W/m²)')
    title(ax, i, m)
    ax.text(0.04, 0.93, f"Spearman $\\rho$ = {vt.loc[m, 'spearman']:.2f}", transform=ax.transAxes, fontsize=7.5)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].set_ylabel('Test MAE (W/m²)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=7, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.1),
           markerscale=1.8)
fig.tight_layout()
save(fig, 'f4_scatter_val_vs_test')

# ── f5: hiperparámetros ganadores ─────────────────────────────────────────────
hp = [('hp_units_1', 'Units (layer 1)', False), ('hp_units_2', 'Units (layer 2)', False),
      ('hp_dropout_rate', 'Dropout rate', False), ('hp_learning_rate', 'Learning rate', True),
      ('hp_batch_size', 'Batch size', False)]
if all(c in b100 for c, _, _ in hp):
    tuned = b100[b100.optimizer != 'DEFAULT']
    fig, axes = plt.subplots(len(hp), 3, figsize=(7.2, 7.4), sharey='row')
    rng = np.random.default_rng(0)
    for r, (c, name, logy) in enumerate(hp):
        for j, m in enumerate(MODELS):
            ax = axes[r, j]
            for k, o in enumerate(OPTS):
                v = tuned[(tuned.model == m) & (tuned.optimizer == o)][c].values
                ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=4, color=COL[o], alpha=0.6, lw=0)
                if len(v):
                    ax.hlines(np.median(v), k - 0.3, k + 0.3, color='k', lw=1)
            if logy:
                ax.set_yscale('log')
            ax.set_xticks(range(7))
            ax.set_xticklabels([lab(o) for o in OPTS] if r == len(hp) - 1 else [], rotation=45,
                               ha='right', fontsize=7)
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
            if r == 0:
                ax.set_title(m, fontsize=9, fontweight='bold')
            ax.text(0.01, 0.97, f'({PANEL[r * 3 + j] if r * 3 + j < 10 else chr(97 + r * 3 + j)})',
                    transform=ax.transAxes, fontsize=7.5, fontweight='bold', va='top')
            ax.grid(axis='y', alpha=0.25, lw=0.5)
    fig.tight_layout()
    save(fig, 'f5_hp_distributions')

# ── f7: importancia de hiperparámetros ────────────────────────────────────────
imp = pd.read_csv(os.path.join(A, 'hp_importance_rs.csv'))
names = {'units_1': 'Units (layer 1)', 'units_2': 'Units (layer 2)', 'dropout_rate': 'Dropout rate',
         'log10_lr': 'Learning rate', 'batch_size': 'Batch size'}
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharex=True)
for i, m in enumerate(MODELS):
    ax = axes[i]
    g = imp[imp.model == m].set_index('hyperparameter').reindex(list(names))
    ax.barh([names[h] for h in g.index], 100 * g['relative'], color='#4477AA')
    ax.invert_yaxis()
    ax.set_xlabel('Relative importance (%)')
    title(ax, i, f'{m} (n = {int(g.n_evals.iloc[0]):,})')
    if i:
        ax.set_yticklabels([])
    ax.grid(axis='x', alpha=0.25, lw=0.5)
save(fig, 'f7_hp_importance')

# ── f8: equivalencia (IC 90 % de las diferencias pareadas) ────────────────────
pw = pd.read_csv(os.path.join(A, 'pairwise.csv'))
pw = pw[(pw.metric == 'all_mae') & (pw.budget == 'B100')]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
for i, m in enumerate(MODELS):
    ax = axes[i]
    g = pw[pw.model == m].reset_index(drop=True)
    y = np.arange(len(g))[::-1]
    ax.axvspan(-C.TOST_DELTA, C.TOST_DELTA, color='#88CCEE', alpha=0.25, lw=0)
    for d in C.TOST_DELTA_SENS:
        for sgn in (-1, 1):
            ax.axvline(sgn * d, color='0.6', lw=0.6, ls=':')
    ax.axvline(0, color='k', lw=0.6)
    for yy, (_, r) in zip(y, g.iterrows()):
        ok = r[f'equiv_d{C.TOST_DELTA:g}']
        ax.plot([r.ci90_lo, r.ci90_hi], [yy, yy], color='#004488' if ok else '#BB5566', lw=1.4)
        ax.plot(r.mean_diff, yy, 'o', ms=2.8, color='#004488' if ok else '#BB5566')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{lab(a)} − {lab(b)}' for a, b in zip(g.A, g.B)], fontsize=6.3)
    title(ax, i, m)
    ax.set_xlim(-2.2, 2.2)
fig.supxlabel('Paired difference in test MAE (W/m²), mean and 90% BCa interval', fontsize=8.5, y=0.01)
fig.tight_layout()
save(fig, 'f8_equivalence')

# ── f9: estudio de una sola corrida ───────────────────────────────────────────
so = pd.read_csv(os.path.join(A, 'single_run_overall.csv')).set_index('model').reindex(MODELS)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axes[0]
x = np.arange(len(MODELS)); w = 0.11
for k, o in enumerate(OPTS):
    ax.bar(x + (k - 3) * w, 100 * so[f'win_share_{o}'], w, color=COL[o], label=lab(o))
ax.axhline(100 / 7, color='k', ls='--', lw=0.8)
ax.text(-0.45, 100 / 7 + 0.6, 'chance (1/7)', fontsize=6.5, ha='left')
ax.set_xticks(x); ax.set_xticklabels(MODELS)
ax.set_ylabel('Times declared best (%)')
title(ax, 0, 'Apparent winner in a single-run study')
ax.legend(ncol=4, fontsize=6.3, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.0))
ax.set_ylim(0, max(40, 100 * so[[f'win_share_{o}' for o in OPTS]].values.max() + 12))
ax = axes[1]
ax.bar(x - 0.18, so.mean_spread_30seed, 0.34, color='#4477AA', label='30-seed means')
ax.bar(x + 0.18, so.median_spread_single_run, 0.34, color='#CCBB44', label='Single run (median)')
ax.axhline(C.TOST_DELTA, color='k', ls='--', lw=0.8)
ax.text(-0.45, C.TOST_DELTA + 0.05, '$\\delta$', fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(MODELS)
ax.set_ylabel('Best-to-worst gap (W/m²)')
title(ax, 1, 'Gap between best and worst optimizer')
ax.legend(fontsize=7, frameon=False, loc='upper left')
ax.set_ylim(0, max(so.median_spread_single_run) * 1.35)
fig.tight_layout()
save(fig, 'f9_single_run')
