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

# ── f2: ajustados vs. por defecto vs. referencias (eje partido) ──────────────
MK = {'PSO': 'o', 'GA': 's', 'ABC': '^', 'CMAES': 'D', 'TPE': 'v', 'BOHB': 'P', 'RS': 'X', 'DEFAULT': 'o'}
fig = plt.figure(figsize=(7.2, 3.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.08, wspace=0.28)
for i, (col, blk, name) in enumerate((('all_mae', 'all', 'All hours'),
                                      ('day_ghi50_mae', 'day_ghi50', 'Daytime (GHI > 50 W/m²)'))):
    top, bot = fig.add_subplot(gs[0, i]), fig.add_subplot(gs[1, i])
    sp = refs.query(f"reference=='smart_persistence' and split=='test' and block=='{blk}'").mae.iloc[0]
    x = np.arange(len(MODELS)); w = 0.1
    tuned_hi, def_lo, def_hi = 0, 1e9, 0
    for k, o in enumerate(OPTS + ['DEFAULT']):
        g = b100[b100.optimizer == o].groupby('model')[col].agg(['mean', 'std']).reindex(MODELS)
        xs = x + (k - 3.5) * w
        ax = top if o == 'DEFAULT' else bot
        ax.errorbar(x if o == 'DEFAULT' else xs, g['mean'], yerr=g['std'], fmt=MK[o], ms=4 if o != 'DEFAULT' else 5,
                    color=COL[o] if o != 'DEFAULT' else '0.35', mfc=COL[o] if o != 'DEFAULT' else 'white',
                    lw=0.9, capsize=1.5, label=lab(o))
        if o == 'DEFAULT':
            def_lo, def_hi = (g['mean'] - g['std']).min(), (g['mean'] + g['std']).max()
        else:
            tuned_hi = max(tuned_hi, (g['mean'] + g['std']).max())
    lo_b = b100[b100.optimizer != 'DEFAULT'].groupby('model')[col].agg(['mean', 'std'])
    bot_lo = (lo_b['mean'] - lo_b['std']).min() - 0.3
    bot_hi = max(tuned_hi, sp) + 0.4
    bot.set_ylim(bot_lo, bot_hi); top.set_ylim(def_lo - 1.5, def_hi + 1.5)
    bot.axhline(sp, color='k', ls='--', lw=0.9, label='Smart persistence')
    top.spines['bottom'].set_visible(False); bot.spines['top'].set_visible(False)
    top.tick_params(bottom=False, labelbottom=False)
    d = 0.012
    for ax, yy in ((top, 0), (bot, 1)):
        ax.plot((-d, +d), (yy - d * 1.6, yy + d * 1.6), transform=ax.transAxes, color='k', clip_on=False, lw=0.8)
    bot.set_xticks(x); bot.set_xticklabels(MODELS)
    bot.set_ylabel('Test MAE (W/m²)'); bot.yaxis.set_label_coords(-0.12, 0.8)
    title(top, i, name)
    for ax in (top, bot):
        ax.grid(axis='y', alpha=0.25, lw=0.5)
    top.set_xlim(bot.get_xlim()); top.set_xticks(x)
h, l = bot.get_legend_handles_labels()
h2, l2 = top.get_legend_handles_labels()
fig.legend(h + h2, l + l2, loc='lower center', ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.1))
save(fig, 'f2_reference_comparison')

# ── f3: presupuestos bajos (ejes independientes) ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
bud = [10, 25, 50, 100]
for i, m in enumerate(MODELS):
    ax = axes[i]
    for o in OPTS:
        g = core[(core.model == m) & (core.optimizer == o)]
        st = g.groupby('budget').all_mae.agg(['mean', 'std', 'count'])
        st = st.reindex([f'B{b}' for b in bud])
        half = 1.96 * st['std'] / np.sqrt(st['count'])
        off = (OPTS.index(o) - 3) * 0.035
        xs = np.array(bud) * (1 + off)
        ax.errorbar(xs, st['mean'], yerr=half, color=COL[o], lw=1, marker=MK[o], ms=3.5,
                    capsize=1.2, elinewidth=0.6, label=lab(o))
    ax.set_xscale('log'); ax.set_xticks(bud); ax.set_xticklabels(bud)
    ax.minorticks_off()
    ax.set_xlabel('Evaluations ($b$)')
    title(ax, i, m)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].set_ylabel('Test MAE (W/m²)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=7, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout(rect=(0, 0.04, 1, 1))
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
        # no equivalente: color, línea discontinua y rombo abierto (no solo color)
        ax.plot([r.ci90_lo, r.ci90_hi], [yy, yy], color='#004488' if ok else '#BB5566', lw=1.4,
                ls='-' if ok else (0, (3, 1.5)))
        ax.plot(r.mean_diff, yy, 'o' if ok else 'D', ms=2.8 if ok else 4.2,
                color='#004488' if ok else '#BB5566', mfc='#004488' if ok else 'white', mew=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{lab(a)} − {lab(b)}' for a, b in zip(g.A, g.B)], fontsize=6.3)
    title(ax, i, m)
    ax.set_xlim(-2.2, 2.2)
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

# ── f10: diseños de la literatura (estudios emulados con semillas disjuntas) ──
LD = os.path.join(A, 'literature_designs.csv')
if os.path.exists(LD):
    ld = pd.read_csv(LD)
    MC = {'MLP': '#BB5566', 'LSTM': '#004488', 'GRU': '#DDAA33'}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for m in MODELS:
        g = ld[ld.model == m].sort_values('n_runs')
        axes[0].plot(g.n_runs, 100 * g.p_true_best, 'o-', ms=3, color=MC[m], label=m)
        axes[1].plot(g.n_runs, 100 * g.p_same_winner, 'o-', ms=3, color=MC[m], label=m)
        h = g[g.n_runs >= 3]
        axes[2].plot(h.n_runs, 100 * h.p_prop, 'o-', ms=3, color=MC[m], label=f'{m}')
        axes[2].plot(h.n_runs, 100 * h.p_prop_holm, 's--', ms=2.5, color=MC[m], lw=0.9)
        axes[2].plot(h.n_runs, 100 * h.p_prop_delta, '^:', ms=2.5, color=MC[m], lw=0.9)
    # nivel esperado por azar con el mismo diseño (optimizadores intercambiables)
    if 'p_true_best_null' in ld:
        for ax, c in zip(axes, ('p_true_best_null', 'p_same_winner_null', 'p_prop_null')):
            z = ld[ld.n_runs >= (3 if c == 'p_prop_null' else 1)].groupby('n_runs')[c].agg(['min', 'max'])
            ax.fill_between(z.index, 100 * z['min'], 100 * z['max'], color='0.6', alpha=0.35, lw=0, zorder=0)
    for i, (ax, t, yl) in enumerate(zip(axes, ['Winner is the true best', 'Two studies agree on winner',
                                              'Proposed method "wins"'],
                                         ['Studies (%)'] * 3)):
        title(ax, i, t)
        ax.set_xlabel('Runs per optimizer ($n$)')
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel('Studies (%)')
    axes[0].legend(fontsize=7, frameon=False, loc='upper left')
    from matplotlib.lines import Line2D
    axes[2].legend([Line2D([], [], color='0.3', ls='-', marker='o', ms=3),
                    Line2D([], [], color='0.3', ls='--', marker='s', ms=2.5),
                    Line2D([], [], color='0.3', ls=':', marker='^', ms=2.5)],
                   ['uncorrected', 'Holm', 'uncorrected, gap > $\\delta$'], fontsize=6.3, frameon=False,
                   loc='upper left')
    fig.tight_layout()
    save(fig, 'f10_literature_designs')

# ── f11: fidelidad del benchmark sustituto ───────────────────────────────────
SR = os.path.join(A, '..', 'surrogate', 'replay_vs_real.csv')
if os.path.exists(SR):
    sr = pd.read_csv(SR)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    mk = {10: 'o', 25: 's', 50: '^', 100: 'D'}
    for b, g in sr.groupby('eval'):
        for o in [o for o in OPTS if o in set(sr.optimizer)]:
            h = g[g.optimizer == o]
            ax.scatter(h.real, h.surrogate, s=14, marker=mk[b], color=COL[o], lw=0,
                       label=lab(o) if b == 100 else None)
    lo, hi = sr[['real', 'surrogate']].min().min() - 0.3, sr[['real', 'surrogate']].max().max() + 0.3
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.7)
    ax.set_xlabel('Experiment: best validation MAE (W/m²)')
    ax.set_ylabel('Surrogate: best validation MAE (W/m²)')
    title(ax, 0, 'Replay at $b$ = 10, 25, 50, 100')
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc='upper left', markerscale=1.2)
    ax.grid(alpha=0.25, lw=0.5)
    ax = axes[1]
    g = sr[sr['eval'] == 100]
    x = np.arange(len(MODELS)); w = 0.12
    for k, o in enumerate([o for o in OPTS if o in set(g.optimizer)]):
        h = g[g.optimizer == o].set_index('model').reindex(MODELS)
        xs = x + (k - 2.5) * w
        ax.scatter(xs, h.real, color=COL[o], s=16, zorder=3)
        ax.scatter(xs, h.surrogate, facecolor='white', edgecolor=COL[o], s=16, zorder=3)
        ax.vlines(xs, h.real, h.surrogate, color=COL[o], lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(MODELS)
    ax.set_ylabel('Best validation MAE at $b$ = 100 (W/m²)')
    title(ax, 1, 'Experiment (filled) vs. surrogate (open)')
    ax.grid(axis='y', alpha=0.25, lw=0.5)
    fig.tight_layout()
    save(fig, 'f11_surrogate')
