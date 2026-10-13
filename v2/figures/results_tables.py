"""
Tablas LaTeX y números clave del manuscrito, generados de los CSV de analyze.py.

    python figures/results_tables.py [carpeta_de_analisis]

Escribe en la carpeta de análisis:
  tables.tex    tablas listas para pegar (resultados, estadística, varianza,
                una sola corrida, presupuestos bajos, población, tiempo)
  numbers.json  todos los números citados en el texto
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

A = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.OUT_DIR, 'analysis')
OPTS = ['PSO', 'GA', 'ABC', 'CMAES', 'TPE', 'BOHB', 'RS']
LBL = {'CMAES': 'CMA-ES', 'DEFAULT': 'Default'}
lab = lambda o: LBL.get(o, o)
MODELS = C.MODELS
rd = lambda f: pd.read_csv(os.path.join(A, f))

runs = rd('runs_long.csv')
core = runs[runs.tag == 'core']
b100 = core[core.budget == 'B100']
refs = rd('reference_forecasts.csv')
pw = rd('pairwise.csv')
fr = rd('friedman.csv')
var = rd('variance_decomposition.csv')
so = rd('single_run_overall.csv').set_index('model')
sp = rd('single_run_pairs.csv')
gain = rd('hpo_gain_vs_default.csv')
vt = rd('val_test_correlation.csv').set_index('model')
pop = rd('popsens.csv')
imp = rd('hp_importance_rs.csv')
symb = json.load(open(os.path.join(A, 'symbolic_refit.json'))) if os.path.exists(
    os.path.join(A, 'symbolic_refit.json')) else None

N = {}
f2 = lambda x: f'{x:.2f}'
f1 = lambda x: f'{x:.1f}'
pm = lambda m, s, f=f2: f'${f(m)} \\pm {f(s)}$'
out = []

# ── Tabla de resultados principal ─────────────────────────────────────────────
roll_bad_thr = 25.0          # W/m²; MAE combinado de rolling-origin claramente anómalo
rows = []
for m in MODELS:
    for o in OPTS + ['DEFAULT']:
        g = b100[(b100.model == m) & (b100.optimizer == o)]
        if g.empty:
            continue
        rows.append((m, o, g))
lines = [r'\begin{tabular}{llccccc}', r'\toprule',
         r'\textbf{Model} & \textbf{Optimizer} & \textbf{MAE} & \textbf{RMSE} & \textbf{Daytime MAE} & '
         r'$\boldsymbol{S_{\mathrm{smart}}}$ \textbf{(\%)} & \textbf{Rolling MAE} \\', r'\midrule']
for k, (m, o, g) in enumerate(rows):
    best = b100[(b100.model == m) & (b100.optimizer != 'DEFAULT')].groupby('optimizer').all_mae.mean().idxmin()
    mae = pm(g.all_mae.mean(), g.all_mae.std())
    if o == best:
        mae = r'$\mathbf{' + f2(g.all_mae.mean()) + r'} \pm ' + f2(g.all_mae.std()) + '$'
    roll = f'{g.roll_mae.median():.2f}'
    nbad = int((g.roll_mae > roll_bad_thr).sum())
    if nbad and o != 'DEFAULT':
        roll += f'$^{{({nbad})}}$'
    lines.append(f"{m if (k == 0 or rows[k - 1][0] != m) else ''} & {lab(o)} & {mae} & "
                 f"{pm(g.all_rmse.mean(), g.all_rmse.std())} & {pm(g.day_ghi50_mae.mean(), g.day_ghi50_mae.std())} & "
                 f"{100 * g.all_skill_rmse_smart.mean():.1f} & {roll} \\\\")
    if o == 'DEFAULT' and m != MODELS[-1]:
        lines.append(r'\midrule')
lines.append(r'\midrule')
for ref, name in (('persistence', 'Persistence'), ('smart_persistence', 'Smart persistence')):
    t = refs.query(f"reference=='{ref}' and split=='test'").set_index('block')
    rl = refs[(refs.reference == ref) & refs.split.str.startswith('roll') & (refs.block == 'all')]
    lines.append(f"\\multicolumn{{2}}{{l}}{{{name}}} & {f2(t.loc['all','mae'])} & {f2(t.loc['all','rmse'])} & "
                 f"{f2(t.loc['day_ghi50','mae'])} & -- & -- \\\\")
if symb:
    lines.append(f"\\multicolumn{{2}}{{l}}{{Symbolic (Eq.~\\ref{{eq:symbolic}})}} & {f2(symb['test_all']['mae'])} & "
                 f"{f2(symb['test_all']['rmse'])} & {f2(symb['test_day_ghi50']['mae'])} & -- & -- \\\\")
lines += [r'\bottomrule', r'\end{tabular}']
out.append(('tab:results_main', '\n'.join(lines)))

# ── Estadística por arquitectura ──────────────────────────────────────────────
main = pw[(pw.metric == 'all_mae') & (pw.budget == 'B100')]
fm = fr[(fr.metric == 'all_mae') & (fr.budget == 'B100')].set_index('model')
lines = [r'\begin{tabular}{lccccccccc}', r'\toprule',
         r'\textbf{Model} & $\chi^2_F$ & $p$ & \textbf{Best rank} & \textbf{Sig.} & '
         r'\multicolumn{3}{c}{\textbf{Equivalent pairs}} & $\boldsymbol{\Delta_{\min}}$ & $|\hat{A}_{12}-0.5|$ \\',
         r'\cmidrule(lr){6-8}', r' & & & & \textbf{(Holm)} & $\delta=0.5$ & $\delta=1$ & $\delta=2$ & \textbf{(median)} & \textbf{(max)} \\',
         r'\midrule']
for m in MODELS:
    g = main[main.model == m]
    f = fm.loc[m]
    ranks = {o: f[f'rank_{o}'] for o in OPTS if f'rank_{o}' in f}
    bo = min(ranks, key=ranks.get)
    lines.append(f"{m} & {f.chi2:.2f} & {f.p:.3f} & {lab(bo)} ({ranks[bo]:.2f}) & {int(g.significant.sum())}/{len(g)} & "
                 f"{int(g['equiv_d0.5'].sum())} & {int(g['equiv_d1'].sum())} & {int(g['equiv_d2'].sum())} & "
                 f"{g.mde_80.median():.2f} & {np.abs(g.A12 - 0.5).max():.2f} \\\\")
    N[f'{m}_friedman_chi2'] = round(float(f.chi2), 2); N[f'{m}_friedman_p'] = round(float(f.p), 3)
    N[f'{m}_sig_pairs'] = int(g.significant.sum()); N[f'{m}_n_pairs'] = len(g)
    for d in ('0.5', '1', '2'):
        N[f'{m}_equiv_d{d}'] = int(g[f'equiv_d{d}'].sum())
    N[f'{m}_mde_median'] = round(float(g.mde_80.median()), 2)
    N[f'{m}_mde_range'] = [round(float(g.mde_80.min()), 2), round(float(g.mde_80.max()), 2)]
    N[f'{m}_max_abs_diff'] = round(float(g.mean_diff.abs().max()), 2)
    N[f'{m}_A12_maxdev'] = round(float(np.abs(g.A12 - 0.5).max()), 3)
    N[f'{m}_ranks'] = {o: round(float(v), 2) for o, v in ranks.items()}
    ne = g[~g['equiv_d1']]
    N[f'{m}_nonequiv_pairs'] = [f'{a}-{b}: {d:.2f} [{l:.2f}, {h:.2f}]' for a, b, d, l, h in
                                zip(ne.A, ne.B, ne.mean_diff, ne.ci90_lo, ne.ci90_hi)]
    sg = g[g.significant]
    N[f'{m}_sig_list'] = [f'{a}-{b}: {d:.2f}, p_holm={p:.3f}' for a, b, d, p in zip(sg.A, sg.B, sg.mean_diff, sg.p_holm)]
lines += [r'\bottomrule', r'\end{tabular}']
out.append(('tab:stat_tests', '\n'.join(lines)))
N['total_equiv_d1'] = int(main['equiv_d1'].sum()); N['total_pairs'] = len(main)
N['total_sig'] = int(main.significant.sum())

# ── Descomposición de varianza ────────────────────────────────────────────────
lines = [r'\begin{tabular}{lcc}', r'\toprule', r'\textbf{Term} & \textbf{Sum of squares} & $\boldsymbol{\eta^2}$ \textbf{(\%)} \\', r'\midrule']
names = {'architecture': 'Architecture', 'optimizer': 'Optimizer', 'architecture x optimizer': 'Architecture $\\times$ optimizer',
         'seed (block)': 'Seed (block)', 'residual': 'Residual'}
for _, r in var.iterrows():
    lines.append(f"{names.get(r.term, r.term)} & {r.SS:.1f} & {100 * r.eta2:.1f} \\\\")
    N[f"eta2_{r.term.replace(' ', '_')}"] = round(100 * float(r.eta2), 1)
lines += [r'\bottomrule', r'\end{tabular}']
out.append(('tab:variance', '\n'.join(lines)))

# ── Fiabilidad de una sola corrida ────────────────────────────────────────────
lines = [r'\begin{tabular}{lcccccc}', r'\toprule',
         r'\textbf{Model} & \textbf{Best by mean} & $P(\text{best})$ & \textbf{Gap, 30 seeds} & \textbf{Gap, single run} & '
         r'$P(\text{gap}>\delta)$ & $n_{\text{TOST}}$ \\', r' & & \textbf{(\%)} & \textbf{(W/m$^2$)} & \textbf{(median, W/m$^2$)} & \textbf{(\%)} & \textbf{(median)} \\',
         r'\midrule']
for m in MODELS:
    r = so.loc[m]; q = sp[sp.model == m]
    lines.append(f"{m} & {lab(r.best_by_mean)} & {100 * r.p_single_run_picks_best:.1f} & {r.mean_spread_30seed:.2f} & "
                 f"{r.median_spread_single_run:.2f} & {100 * r.p_spread_gt_delta:.1f} & {int(q.n_req_tost.median())} \\\\")
    N[f'{m}_sr_p_best'] = round(100 * float(r.p_single_run_picks_best), 1)
    N[f'{m}_sr_gap30'] = round(float(r.mean_spread_30seed), 2)
    N[f'{m}_sr_gap1'] = round(float(r.median_spread_single_run), 2)
    N[f'{m}_sr_p_gap'] = round(100 * float(r.p_spread_gt_delta), 1)
    N[f'{m}_sr_pair_p_gap_median'] = round(100 * float(q.p_gap_gt_delta.median()), 1)
    N[f'{m}_sr_pair_p_gap_range'] = [round(100 * float(q.p_gap_gt_delta.min()), 1), round(100 * float(q.p_gap_gt_delta.max()), 1)]
    N[f'{m}_sr_n_req_median'] = int(q.n_req_tost.median()); N[f'{m}_sr_n_req_max'] = int(q.n_req_tost.max())
    N[f'{m}_sr_best'] = r.best_by_mean
lines += [r'\bottomrule', r'\end{tabular}']
out.append(('tab:single_run', '\n'.join(lines)))

# ── Presupuestos bajos ────────────────────────────────────────────────────────
lines = [r'\begin{tabular}{lccccccccc}', r'\toprule',
         r' & \multicolumn{3}{c}{\textbf{MLP}} & \multicolumn{3}{c}{\textbf{LSTM}} & \multicolumn{3}{c}{\textbf{GRU}} \\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
         r'$b$ & \textbf{Range} & \textbf{Sig.} & \textbf{Eq.} & \textbf{Range} & \textbf{Sig.} & \textbf{Eq.} & \textbf{Range} & \textbf{Sig.} & \textbf{Eq.} \\',
         r'\midrule']
for b in [10, 25, 50, 100]:
    cells = [str(b)]
    for m in MODELS:
        g = pw[(pw.metric == 'all_mae') & (pw.budget == f'B{b}') & (pw.model == m)]
        mm = core[(core.model == m) & (core.budget == f'B{b}') & (core.optimizer != 'DEFAULT')].groupby('optimizer').all_mae.mean()
        cells += [f'{mm.min():.2f}--{mm.max():.2f}', f'{int(g.significant.sum())}', f"{int(g['equiv_d1'].sum())}"]
        N[f'{m}_B{b}_range'] = [round(float(mm.min()), 2), round(float(mm.max()), 2)]
        N[f'{m}_B{b}_best'] = mm.idxmin(); N[f'{m}_B{b}_worst'] = mm.idxmax()
        N[f'{m}_B{b}_sig'] = int(g.significant.sum()); N[f'{m}_B{b}_eq'] = int(g['equiv_d1'].sum())
        N[f'{m}_B{b}_sig_list'] = [f'{a}-{c}: {d:.2f}' for a, c, d in zip(g[g.significant].A, g[g.significant].B, g[g.significant].mean_diff)]
    lines.append(' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
out.append(('tab:low_budget', '\n'.join(lines)))

# ── Población y tiempo ────────────────────────────────────────────────────────
lines = [r'\begin{tabular}{lccc}', r'\toprule', r'\textbf{Optimizer} & $n=5$ & $n=10$ & $n=20$ \\', r'\midrule']
for o in C.POP_SENS_OPTIMIZERS:
    g = pop[pop.optimizer == o].set_index('pop')
    if C.POP_SIZE not in g.index:          # población por defecto: corridas principales del MLP
        h = b100[(b100.model == 'MLP') & (b100.optimizer == o)].all_mae
        g.loc[C.POP_SIZE, ['mae_mean', 'mae_sd', 'n']] = [h.mean(), h.std(), len(h)]
    g = g.sort_index()
    cells = [pm(g.loc[p, 'mae_mean'], g.loc[p, 'mae_sd']) if p in g.index else '--' for p in (5, 10, 20)]
    lines.append(f"{o} & " + ' & '.join(cells) + r' \\')
    N[f'pop_{o}'] = {int(p): round(float(g.loc[p, 'mae_mean']), 2) for p in g.index}
    N[f'pop_{o}_sd'] = {int(p): round(float(g.loc[p, 'mae_sd']), 2) for p in g.index}
    N[f'pop_{o}_n'] = {int(p): int(g.loc[p, 'n']) for p in g.index}
lines += [r'\bottomrule', r'\end{tabular}']
out.append(('tab:popsens', '\n'.join(lines)))

tuned = b100[b100.optimizer != 'DEFAULT']
lines = [r'\begin{tabular}{lccccccc}', r'\toprule', r'\textbf{Model} & ' + ' & '.join(f'\\textbf{{{lab(o)}}}' for o in OPTS) + r' \\', r'\midrule']
for m in MODELS:
    t = tuned[tuned.model == m].groupby('optimizer').hpo_time_s.mean() / 60
    lines.append(f'{m} & ' + ' & '.join(f'{t[o]:.0f}' for o in OPTS) + r' \\')
    N[f'{m}_hpo_min'] = {o: round(float(t[o]), 1) for o in OPTS}
lines.append(r'\midrule')
for m in MODELS:
    e = tuned[tuned.model == m].groupby('optimizer').hpo_epochs.mean()
    lines.append(f'{m} (epochs) & ' + ' & '.join(f'{e[o]:.0f}' for o in OPTS) + r' \\')
    N[f'{m}_hpo_epochs'] = {o: round(float(e[o]), 0) for o in OPTS}
bo = tuned[tuned.optimizer == 'BOHB']
N['bohb_configs'] = [int(bo.n_evals.min()), int(bo.n_evals.max()), round(float(bo.n_evals.mean()), 1)]
N['bohb_pruned_mean'] = round(float(bo.n_pruned.mean()), 1) if 'n_pruned' in bo else None
lines += [r'\bottomrule', r'\end{tabular}']
out.append(('tab:opt_time', '\n'.join(lines)))

# ── Arquitecturas (comparación directa) ──────────────────────────────────────
ap = rd('architecture_pairwise.csv')
lines = [r'\begin{tabular}{lccc}', r'\toprule',
         r'\textbf{Optimizer} & \textbf{LSTM $-$ MLP} & \textbf{GRU $-$ MLP} & \textbf{LSTM $-$ GRU} \\', r'\midrule']
for o_ in OPTS + ['DEFAULT']:
    g = ap[ap.optimizer == o_]
    cells = []
    for a, b in (('LSTM', 'MLP'), ('GRU', 'MLP'), ('LSTM', 'GRU')):
        r = g[(g.A == a) & (g.B == b)]
        sign = 1
        if r.empty:
            r = g[(g.A == b) & (g.B == a)]; sign = -1
        if r.empty:
            cells.append('--'); continue
        r = r.iloc[0]
        d, lo, hi = sign * r.mean_diff, *(sorted([sign * r.ci95_lo, sign * r.ci95_hi]))
        mark = '$^{*}$' if r.significant else ''
        cells.append(f'${d:.2f}$ [${lo:.2f}$, ${hi:.2f}$]{mark}')
        N[f'arch_{o_}_{a}-{b}'] = [round(float(d), 2), round(float(lo), 2), round(float(hi), 2), bool(r.significant),
                                   bool(r.equiv_d1)]
    lines.append(f'{lab(o_)} & ' + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
out.append(('tab:architectures', '\n'.join(lines)))

# ── Números sueltos ───────────────────────────────────────────────────────────
for m in MODELS:
    g = tuned[tuned.model == m]
    N[f'{m}_tuned_mean_range'] = [round(float(g.groupby('optimizer').all_mae.mean().min()), 2),
                                  round(float(g.groupby('optimizer').all_mae.mean().max()), 2)]
    N[f'{m}_tuned_day_range'] = [round(float(g.groupby('optimizer').day_ghi50_mae.mean().min()), 2),
                                 round(float(g.groupby('optimizer').day_ghi50_mae.mean().max()), 2)]
    N[f'{m}_seed_sd_range'] = [round(float(g.groupby('optimizer').all_mae.std().min()), 2),
                               round(float(g.groupby('optimizer').all_mae.std().max()), 2)]
    N[f'{m}_skill_smart_range'] = [round(100 * float(g.groupby('optimizer').all_skill_rmse_smart.mean().min()), 1),
                                   round(100 * float(g.groupby('optimizer').all_skill_rmse_smart.mean().max()), 1)]
    N[f'{m}_skill_pers_range'] = [round(100 * float(g.groupby('optimizer').all_skill_rmse_pers.mean().min()), 1),
                                  round(100 * float(g.groupby('optimizer').all_skill_rmse_pers.mean().max()), 1)]
    N[f'{m}_frac_runs_beat_smart_mae'] = round(100 * float((g.all_mae < refs.query(
        "reference=='smart_persistence' and split=='test' and block=='all'").mae.iloc[0]).mean()), 1)
    d = b100[(b100.model == m) & (b100.optimizer == 'DEFAULT')]
    N[f'{m}_default'] = [round(float(d.all_mae.mean()), 2), round(float(d.all_mae.std()), 2)]
    N[f'{m}_default_day'] = round(float(d.day_ghi50_mae.mean()), 2)
    N[f'{m}_default_skill_smart'] = round(100 * float(d.all_skill_rmse_smart.mean()), 1)
    gg = gain[gain.model == m]
    N[f'{m}_gain_pct_range'] = [round(float(gg.reduction_pct_mean.min()), 1), round(float(gg.reduction_pct_mean.max()), 1)]
    N[f'{m}_val_test'] = [round(float(vt.loc[m, 'pearson']), 2), round(float(vt.loc[m, 'spearman']), 2)]
    ii = imp[imp.model == m].set_index('hyperparameter').relative
    N[f'{m}_hp_importance'] = {k: round(100 * float(v), 1) for k, v in ii.items()}
    rr = g.roll_mae
    N[f'{m}_roll_median_range'] = [round(float(g.groupby('optimizer').roll_mae.median().min()), 2),
                                   round(float(g.groupby('optimizer').roll_mae.median().max()), 2)]
    N[f'{m}_roll_bad_runs'] = int((rr > roll_bad_thr).sum())
    q = g[[f'roll_q{i}_mae' for i in range(1, 5)]]
    N[f'{m}_roll_quarter_median'] = [round(float(x), 2) for x in q.median()]
    N[f'{m}_roll_folds_gt30'] = int((q > 30).sum().sum())
    N[f'{m}_roll_max_fold'] = round(float(q.max().max()), 1)
for met, key in (('roll_mae', 'roll'), ('day_ghi50_mae', 'day')):
    g = pw[(pw.metric == met) & (pw.budget == 'B100')]
    for m in MODELS:
        h = g[g.model == m]
        N[f'{m}_{key}_equiv_d1'] = int(h['equiv_d1'].sum()); N[f'{m}_{key}_sig'] = int(h.significant.sum())
        N[f'{m}_{key}_npairs'] = len(h)
        N[f'{m}_{key}_sig_list'] = [f'{a}-{b}: {d:.2f}' for a, b, d in zip(h[h.significant].A, h[h.significant].B, h[h.significant].mean_diff)]
for m in MODELS:
    g = tuned[tuned.model == m]
    N[f'{m}_rmse_range'] = [round(float(g.groupby('optimizer').all_rmse.mean().min()), 2), round(float(g.groupby('optimizer').all_rmse.mean().max()), 2)]
    N[f'{m}_r2_range'] = [round(float(g.groupby('optimizer').all_r2.mean().min()), 3), round(float(g.groupby('optimizer').all_r2.mean().max()), 3)]
    N[f'{m}_day_r2_range'] = [round(float(g.groupby('optimizer').day_ghi50_r2.mean().min()), 3), round(float(g.groupby('optimizer').day_ghi50_r2.mean().max()), 3)]
    N[f'{m}_mbe_range'] = [round(float(g.groupby('optimizer').all_mbe.mean().min()), 2), round(float(g.groupby('optimizer').all_mbe.mean().max()), 2)]
    N[f'{m}_elev_mae_range'] = [round(float(g.groupby('optimizer').day_elev5_mae.mean().min()), 2), round(float(g.groupby('optimizer').day_elev5_mae.mean().max()), 2)]
ak = [k for k in N if k.startswith('arch_') and 'DEFAULT' not in k]
N['arch_recurrent_vs_mlp_sig'] = sum(N[k][3] for k in ak if k.endswith('-MLP'))
N['arch_recurrent_vs_mlp_n'] = sum(1 for k in ak if k.endswith('-MLP'))
N['arch_recurrent_vs_mlp_range'] = [min(N[k][0] for k in ak if k.endswith('-MLP')), max(N[k][0] for k in ak if k.endswith('-MLP'))]
N['arch_lstm_gru_sig'] = sum(N[k][3] for k in ak if k.endswith('LSTM-GRU'))
N['arch_lstm_gru_equiv'] = sum(N[k][4] for k in ak if k.endswith('LSTM-GRU'))
N['arch_lstm_gru_range'] = [min(N[k][0] for k in ak if k.endswith('LSTM-GRU')), max(N[k][0] for k in ak if k.endswith('LSTM-GRU'))]
N['all_rmse_range'] = [min(N[f'{m}_rmse_range'][0] for m in MODELS), max(N[f'{m}_rmse_range'][1] for m in MODELS)]
N['val_test_all'] = [round(float(vt.loc['all', 'pearson']), 2), round(float(vt.loc['all', 'spearman']), 2)]
for ref in ('persistence', 'smart_persistence'):
    t = refs.query(f"reference=='{ref}' and split=='test'").set_index('block')
    N[f'{ref}_mae'] = round(float(t.loc['all', 'mae']), 2); N[f'{ref}_rmse'] = round(float(t.loc['all', 'rmse']), 2)
    N[f'{ref}_day_mae'] = round(float(t.loc['day_ghi50', 'mae']), 2)
    rl = refs[(refs.reference == ref) & refs.split.str.startswith('roll') & (refs.block == 'all')]
    N[f'{ref}_roll_quarters_mae'] = [round(float(x), 2) for x in rl.sort_values('split').mae]
best = tuned.loc[tuned.all_mae.idxmin()]
N['best_run'] = f"{best.run}: {best.all_mae:.2f}"
N['n_runs'] = runs[runs.budget == 'B100'].groupby('tag').size().to_dict()
pd_ = runs[(runs.tag == 'popsens') & (runs.budget == 'B100')]
N['popsens_n'] = len(pd_)

# ── Diseños de la literatura ──────────────────────────────────────────────────
LD = os.path.join(A, 'literature_designs.csv')
if os.path.exists(LD):
    ld = rd('literature_designs.csv')
    rg = lambda c, n: (100 * ld[ld.n_runs == n][c].min(), 100 * ld[ld.n_runs == n][c].max())
    def fmt(c, n):
        lo_, hi_ = rg(c, n)
        if np.isnan(lo_):
            return '--'
        return f'{lo_:.0f}' if round(lo_) == round(hi_) else f'{lo_:.0f}--{hi_:.0f}'
    lines = [r'\begin{tabular}{cccccc}', r'\toprule',
             r'$\boldsymbol{n}$ & \textbf{Winner is} & \textbf{Same winner} & \multicolumn{3}{c}{\textbf{Proposed method significantly better than a rival}} \\',
             r'\cmidrule(lr){4-6}',
             r' & \textbf{true best} & \textbf{in replication} & \textbf{Uncorrected} & \textbf{Holm} & \textbf{Uncorrected, gap $>\delta$} \\', r'\midrule']
    for n in sorted(ld.n_runs.unique()):
        lines.append(f"{n} & {fmt('p_true_best', n)} & {fmt('p_same_winner', n)} & {fmt('p_prop', n)} & "
                     f"{fmt('p_prop_holm', n)} & {fmt('p_prop_delta', n)} \\\\")
        for c in ('p_true_best', 'p_same_winner', 'p_prop', 'p_prop_holm', 'p_prop_delta', 'p_any_sig', 'p_any_sig_holm'):
            lo_, hi_ = rg(c, n)
            if not np.isnan(lo_):
                N[f'lit_n{n}_{c}'] = [round(float(lo_), 1), round(float(hi_), 1)]
    lines += [r'\bottomrule', r'\end{tabular}']
    out.append(('tab:lit_designs', '\n'.join(lines)))

# ── Benchmark sustituto ───────────────────────────────────────────────────────
SD = os.path.join(A, '..', 'surrogate')
if os.path.exists(os.path.join(SD, 'fidelity.csv')):
    fi = pd.read_csv(os.path.join(SD, 'fidelity.csv')).set_index('model')
    cmp_ = pd.read_csv(os.path.join(SD, 'replay_vs_real.csv'))
    meta = json.load(open(os.path.join(SD, 'replay_meta.json')))
    import joblib
    lines = [r'\begin{tabular}{lccccccc}', r'\toprule',
             r'\textbf{Model} & \textbf{Evaluations} & \textbf{Spearman} & $\boldsymbol{R^2}$ & $\boldsymbol{\kappa}$ & '
             r'\textbf{Diff.} & $\boldsymbol{\rho}$ & \textbf{Last} \\',
             r' & & \textbf{(held-out seeds)} & & & \textbf{(W/m$^2$)} & \textbf{(ranking)} & \textbf{(real / surr.)} \\', r'\midrule']
    for m in MODELS:
        kap = joblib.load(os.path.join(SD, f'{m}.joblib')).get('kappa', np.nan)
        g = cmp_[(cmp_.model == m) & (cmp_['eval'] == 100)]
        gv = g[g.optimizer != 'RS']
        diff = float((gv.real - gv.surrogate).abs().mean())
        rho = stats.spearmanr(g.real, g.surrogate)[0]
        last_r, last_s = g.loc[g.real.idxmax(), 'optimizer'], g.loc[g.surrogate.idxmax(), 'optimizer']
        lines.append(f"{m} & {int(fi.loc[m, 'n_evals']):,} & {fi.loc[m, 'spearman']:.2f} & {fi.loc[m, 'r2']:.2f} & "
                     f"{kap:.2f} & {diff:.2f} & {rho:.2f} & {lab(last_r)} / {lab(last_s)} \\\\".replace(',', '{,}'))
        N[f'sur_{m}'] = {'n': int(fi.loc[m, 'n_evals']), 'spearman': round(float(fi.loc[m, 'spearman']), 2),
                         'r2': round(float(fi.loc[m, 'r2']), 2), 'kappa': round(float(kap), 2),
                         'diff_b100_nonRS': round(diff, 2), 'rho_b100': round(float(rho), 2),
                         'last_real': last_r, 'last_sur': last_s}
    gv = cmp_[cmp_.optimizer != 'RS']
    N['sur_diff_all_nonRS'] = round(float((gv.real - gv.surrogate).abs().mean()), 2)
    N['sur_diff_max_nonRS'] = round(float((gv.real - gv.surrogate).abs().max()), 2)
    N['sur_spearman_all'] = round(float(meta['spearman_all']), 2)
    N['sur_seconds_per_run'] = round(float(meta['seconds_per_run']), 2)
    N['sur_n_evals_total'] = int(fi.n_evals.sum())
    lines += [r'\bottomrule', r'\end{tabular}']
    out.append(('tab:surrogate', '\n'.join(lines)))

with open(os.path.join(A, 'tables.tex'), 'w') as f:
    for k, t in out:
        f.write(f'% ===== {k} =====\n{t}\n\n')
json.dump(N, open(os.path.join(A, 'numbers.json'), 'w'), indent=1, default=str)
print(f'tables.tex y numbers.json en {A}')
