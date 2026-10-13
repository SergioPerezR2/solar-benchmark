"""
Análisis estadístico del benchmark v2 (Sec. 3.6 del manuscrito).

Lee outputs/results/*/*.json y escribe tablas CSV en outputs/analysis/.
Funciona con resultados parciales: cada análisis usa solo las semillas que
tienen TODOS los optimizadores comparados (bloques completos) y reporta n.

    python analyze.py                 # todo
    python analyze.py --boot 2000     # bootstrap más rápido (pruebas)

Todo lo pre-especificado (δ, α, prueba pareada) viene de config.py.
"""
import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import config as C

OUT = os.path.join(C.OUT_DIR, 'analysis')
OPT_ORDER = ['PSO', 'GA', 'ABC', 'CMAES', 'TPE', 'BOHB', 'RS']
BUDGETS = [f'B{b}' for b in C.LOW_BUDGETS + [C.BUDGET]]
FULL = f'B{C.BUDGET}'


# ── Carga ──────────────────────────────────────────────────────────────────────
def load_runs():
    """Una fila por (corrida, presupuesto) con las métricas de test."""
    rows, hist = [], {}
    for path in glob.glob(os.path.join(C.RES_DIR, '*', '*.json')):
        r = json.load(open(path))
        if r.get('dry') or r.get('tag') == 'repro':      # repro: solo para repro_check.py
            continue
        base = {k: r[k] for k in ('run', 'model', 'optimizer', 'seed', 'pop', 'tag')}
        base.update({'best_val_mae': r['best_val_mae'], 'hpo_time_s': r['hpo_time_s'],
                     'final_time_s': r['final_time_s'], 'hpo_epochs': r['total_hpo_epochs'],
                     'n_evals': r['n_evals'], 'n_pruned': r.get('n_pruned', 0)})
        for k, v in (r.get('best_params') or {}).items():
            base[f'hp_{k}'] = v
        fin = r['finals']
        for b, e in fin.items():
            src = fin[e['same_as']] if 'same_as' in e else e
            row = dict(base, budget=b, val_mae_inc=e.get('val_mae_hpo'))
            for block in ('all', 'day_ghi50', 'day_elev5'):
                for k, v in src['test'][block].items():
                    row[f'{block}_{k}'] = v
            if b == FULL and r.get('rolling'):
                for k, v in r['rolling']['pooled']['all'].items():
                    row[f'roll_{k}'] = v
                for f in r['rolling']['folds']:
                    row[f'roll_q{f["fold"] + 1}_mae'] = f['test']['all']['mae']
            rows.append(row)
        hist[(r['tag'], r['run'])] = r['history']
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('No hay resultados todavía.')
    df['optimizer'] = pd.Categorical(df['optimizer'], OPT_ORDER + ['DEFAULT'], ordered=True)
    return df, hist


def references():
    """Persistencia y smart persistence sobre el test y los folds rolling."""
    d = np.load(os.path.join(C.DATA_DIR, 'dataset.npz'))
    out = []
    for name in ('persistence', 'smart_persistence'):
        for split, pos in [('test', d['split_test'])] + [
                (f'roll_q{i + 1}', d[f'fold{i}_test']) for i in range(len(C.ROLLING_FOLDS))]:
            y, p = d['ghi'][pos].astype(float), d[name][pos].astype(float)
            for block, m in (('all', np.ones(len(pos), bool)), ('day_ghi50', d['day_ghi'][pos]),
                             ('day_elev5', d['day_elev'][pos])):
                e = p[m] - y[m]
                out.append({'reference': name, 'split': split, 'block': block,
                            'mae': np.abs(e).mean(), 'rmse': np.sqrt((e ** 2).mean()),
                            'mbe': e.mean(),
                            'r2': 1 - (e ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()})
    return pd.DataFrame(out)


# ── Utilidades estadísticas ────────────────────────────────────────────────────
def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def a12(x, y):
    """P(X < Y) + 0.5 P(X = Y): probabilidad de que X tenga MENOR error que Y."""
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float(((x < y).sum() + 0.5 * (x == y).sum()) / (x.size * y.size))


def a12_label(a):
    d = abs(a - 0.5)
    return 'negligible' if d < 0.06 else 'small' if d < 0.14 else 'medium' if d < 0.21 else 'large'


def boot_ci(d, level, n_boot, seed=0):
    if np.allclose(d, d[0]):
        return float(d[0]), float(d[0])
    res = stats.bootstrap((d,), np.mean, confidence_level=level, n_resamples=n_boot,
                          method='BCa', random_state=seed)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def complete_blocks(sub, opts, metric):
    """Matriz semillas x optimizadores solo con semillas completas."""
    wide = sub.pivot_table(index='seed', columns='optimizer', values=metric, observed=True)
    wide = wide[[o for o in opts if o in wide.columns]]
    # Con resultados parciales, excluir optimizadores con muy pocas semillas
    # para no reducir los bloques completos a casi nada.
    cnt = wide.count()
    wide = wide[[o for o in wide.columns if cnt[o] >= max(3, 0.5 * cnt.max())]]
    return wide.dropna()


# ── Análisis ───────────────────────────────────────────────────────────────────
def summary_table(df):
    core = df[(df.tag == 'core') & (df.budget == FULL)]
    agg = core.groupby(['model', 'optimizer'], observed=True).agg(
        n=('seed', 'count'),
        mae_mean=('all_mae', 'mean'), mae_sd=('all_mae', 'std'),
        rmse_mean=('all_rmse', 'mean'), rmse_sd=('all_rmse', 'std'),
        r2_mean=('all_r2', 'mean'), mbe_mean=('all_mbe', 'mean'),
        day_mae_mean=('day_ghi50_mae', 'mean'), day_mae_sd=('day_ghi50_mae', 'std'),
        day_rmse_mean=('day_ghi50_rmse', 'mean'), day_r2_mean=('day_ghi50_r2', 'mean'),
        skill_smart_mean=('all_skill_rmse_smart', 'mean'),
        skill_pers_mean=('all_skill_rmse_pers', 'mean'),
        roll_mae_mean=('roll_mae', 'mean'), roll_mae_sd=('roll_mae', 'std'),
        hpo_epochs_mean=('hpo_epochs', 'mean'), hpo_time_min=('hpo_time_s', lambda s: s.mean() / 60),
    ).reset_index()
    return agg


def hpo_gain(df):
    """Reducción de MAE frente al baseline por defecto, pareada por semilla."""
    core = df[(df.tag == 'core') & (df.budget == FULL)]
    rows = []
    for m, sub in core.groupby('model'):
        wide = sub.pivot_table(index='seed', columns='optimizer', values='all_mae', observed=True)
        if 'DEFAULT' not in wide:
            continue
        for o in OPT_ORDER:
            if o not in wide:
                continue
            pair = wide[[o, 'DEFAULT']].dropna()
            if len(pair) < 2:
                continue
            red = 100 * (1 - pair[o] / pair['DEFAULT'])
            rows.append({'model': m, 'optimizer': o, 'n': len(pair),
                         'reduction_pct_mean': red.mean(), 'reduction_pct_sd': red.std(),
                         'default_mae_mean': pair['DEFAULT'].mean(), 'tuned_mae_mean': pair[o].mean()})
    return pd.DataFrame(rows)


def friedman_and_pairs(df, metric, budget, n_boot):
    core = df[(df.tag == 'core') & (df.budget == budget) & (df.optimizer != 'DEFAULT')]
    fried, pairs = [], []
    for m, sub in core.groupby('model'):
        wide = complete_blocks(sub, OPT_ORDER, metric)
        n, opts = len(wide), list(wide.columns)
        if n < 3 or len(opts) < 2:
            continue
        if len(opts) >= 3:
            chi, p = stats.friedmanchisquare(*[wide[o] for o in opts])
            ranks = wide.rank(axis=1).mean()
            fried.append({'model': m, 'budget': budget, 'metric': metric, 'n_seeds': n,
                          'k': len(opts), 'chi2': chi, 'p': p,
                          **{f'rank_{o}': ranks[o] for o in opts}})
        block = []
        for a, b in combinations(opts, 2):
            d = (wide[a] - wide[b]).to_numpy()
            try:
                p_w = stats.wilcoxon(d, zero_method='wilcox').pvalue
            except ValueError:
                p_w = 1.0
            lo95, hi95 = boot_ci(d, 0.95, n_boot)
            lo90, hi90 = boot_ci(d, 1 - 2 * C.TOST_ALPHA, n_boot)
            sd = d.std(ddof=1)
            row = {'model': m, 'budget': budget, 'metric': metric, 'n_seeds': n, 'A': a, 'B': b,
                   'mean_A': wide[a].mean(), 'mean_B': wide[b].mean(), 'mean_diff': d.mean(),
                   'ci95_lo': lo95, 'ci95_hi': hi95, 'ci90_lo': lo90, 'ci90_hi': hi90,
                   'p_wilcoxon': p_w, 'A12': a12(wide[a], wide[b]),
                   'sd_diff': sd, 'mde_80': (stats.norm.ppf(1 - C.ALPHA / 2) + stats.norm.ppf(0.8))
                   * sd / np.sqrt(n)}
            row['A12_effect'] = a12_label(row['A12'])
            for delta in [C.TOST_DELTA] + C.TOST_DELTA_SENS:
                row[f'equiv_d{delta:g}'] = bool(-delta < lo90 and hi90 < delta)
            block.append(row)
        adj = holm([r['p_wilcoxon'] for r in block])
        for r, pa in zip(block, adj):
            r['p_holm'] = pa
            r['significant'] = bool(pa < C.ALPHA)
        pairs += block
    return pd.DataFrame(fried), pd.DataFrame(pairs)


def single_run_reliability(df, n_mc=20000, seed=0):
    """¿Qué habría concluido un estudio de UNA sola corrida por optimizador?

    Un estudio así compara una corrida de A contra una de B, cada una con su
    propia semilla. Con todas las combinaciones (i, j) de semillas se estima:
      - p_gap_gt_delta: P(|x_A,i − x_B,j| > δ), es decir, que el estudio vea una
        diferencia mayor que el margen de equivalencia y declare un "ganador";
      - p_reversal: P(signo de x_A,i − x_B,j ≠ signo de la diferencia media con
        todas las semillas), es decir, que elija al ganador contrario;
      - n_req_tost: semillas necesarias para demostrar equivalencia (TOST, 80 %
        de potencia, diferencia real 0) con la σ_d observada.
    Para los k optimizadores a la vez (Monte Carlo, una semilla al azar por
    optimizador) se estima con qué frecuencia el "mejor" del estudio de una
    corrida coincide con el mejor por media y qué brecha mejor–peor aparenta.
    """
    core = df[(df.tag == 'core') & (df.budget == FULL) & (df.optimizer != 'DEFAULT')]
    rng = np.random.default_rng(seed)
    z = stats.norm.ppf(1 - C.TOST_ALPHA) + stats.norm.ppf(0.9)   # β/2 = 0.1
    pairs, overall = [], []
    for m, sub in core.groupby('model'):
        wide = complete_blocks(sub, OPT_ORDER, 'all_mae')
        n, opts = len(wide), list(wide.columns)
        if n < 3 or len(opts) < 2:
            continue
        for a, b in combinations(opts, 2):
            xa, xb = wide[a].to_numpy(), wide[b].to_numpy()
            g = xa[:, None] - xb[None, :]                     # todas las (i, j)
            mean_d = (xa - xb).mean()
            sd_d = (xa - xb).std(ddof=1)
            pairs.append({'model': m, 'n_seeds': n, 'A': a, 'B': b, 'mean_diff': mean_d,
                          'p_gap_gt_delta': float((np.abs(g) > C.TOST_DELTA).mean()),
                          'p_reversal': float((np.sign(g) != np.sign(mean_d)).mean()),
                          'median_abs_gap': float(np.median(np.abs(g))),
                          'sd_diff': sd_d,
                          'n_req_tost': int(np.ceil((z * sd_d / C.TOST_DELTA) ** 2))})
        X = wide.to_numpy()                                   # semillas x optimizadores
        draw = X[rng.integers(0, n, size=(n_mc, len(opts))), np.arange(len(opts))]
        best_mean = int(np.argmin(X.mean(axis=0)))
        spread = draw.max(axis=1) - draw.min(axis=1)
        winners = np.bincount(draw.argmin(axis=1), minlength=len(opts)) / n_mc
        overall.append({'model': m, 'n_seeds': n, 'k': len(opts),
                        'best_by_mean': opts[best_mean],
                        'p_single_run_picks_best': float(winners[best_mean]),
                        'max_single_run_win_share': float(winners.max()),
                        'mean_spread_30seed': float(np.ptp(X.mean(axis=0))),
                        'median_spread_single_run': float(np.median(spread)),
                        'p_spread_gt_delta': float((spread > C.TOST_DELTA).mean()),
                        **{f'win_share_{o}': w for o, w in zip(opts, winners)}})
    return pd.DataFrame(pairs), pd.DataFrame(overall)


def literature_designs(df, sizes=(1, 3, 5, 10, 15, 20, 30), n_rep=2000, seed=1):
    """Emula estudios con n corridas independientes por optimizador.

    Cada estudio remuestrea (con reemplazo) n semillas para cada optimizador de
    forma independiente, como en la literatura. Es un bootstrap plug-in: la
    distribución observada de cada optimizador hace de población, de modo que
    sus medias observadas se toman como verdaderas (con n grande, las tasas de
    detección son una cota superior). Cada cantidad se calcula también bajo la
    hipótesis nula (sufijo _null) con los datos centrados por columna, que
    igualan las medias de los optimizadores y conservan la forma de sus
    distribuciones; así se obtiene la tasa esperada por azar con el mismo diseño.
    Se estima:
      - p_true_best: el optimizador con menor media es el mejor por media de 30 semillas;
      - p_same_winner: dos estudios independientes del mismo diseño eligen el mismo ganador;
      - gap_median: brecha aparente mejor–peor (W/m²);
      - p_any_sig: al menos un par difiere (Mann–Whitney, α = 0.05, sin corrección);
      - p_proposal_wins: un optimizador elegido de antemano resulta significativamente
        mejor que al menos uno de los otros seis (lo que reportaría quien lo "propone").
    """
    core = df[(df.tag == 'core') & (df.budget == FULL) & (df.optimizer != 'DEFAULT')]
    rng = np.random.default_rng(seed)
    rows = []
    for (m, sub), label in [(g, lab) for g in core.groupby('model') for lab in ('', '_null')]:
        X = complete_blocks(sub, OPT_ORDER, 'all_mae').to_numpy()       # semillas x optimizadores
        if label:
            X = X - X.mean(axis=0) + X.mean()                            # H0: medias iguales
        n_s, k = X.shape
        best = int(np.argmin(X.mean(axis=0)))
        for n in sizes:
            win = np.empty((2, n_rep), int); gaps = np.empty(n_rep)
            flags = {f: np.zeros(n_rep, bool) for f in ('any_sig', 'any_sig_holm', 'prop', 'prop_holm', 'prop_delta')}
            pairs = list(combinations(range(k), 2))
            for t in range(2):
                for r in range(n_rep):
                    S = X[rng.integers(0, n_s, size=(n, k)), np.arange(k)]   # n x k
                    mu = S.mean(axis=0)
                    win[t, r] = int(np.argmin(mu))
                    if t or n < 3:
                        continue
                    gaps[r] = mu.max() - mu.min()
                    p_star = rng.integers(k)
                    pv = np.array([stats.mannwhitneyu(S[:, a], S[:, b], alternative='two-sided').pvalue
                                   for a, b in pairs])
                    ph = holm(pv)
                    for (a, b), p, q in zip(pairs, pv, ph):
                        other = b if p_star == a else a
                        mine = p_star in (a, b) and mu[p_star] < mu[other]
                        if p < C.ALPHA:
                            flags['any_sig'][r] = True
                            if mine:
                                flags['prop'][r] = True
                                if mu[other] - mu[p_star] > C.TOST_DELTA:
                                    flags['prop_delta'][r] = True
                        if q < C.ALPHA:
                            flags['any_sig_holm'][r] = True
                            if mine:
                                flags['prop_holm'][r] = True
            if n < 3:
                gaps = np.array([np.ptp(X[rng.integers(0, n_s, size=(n, k)), np.arange(k)].mean(axis=0))
                                 for _ in range(n_rep)])
            row = {'model': m, 'n_runs': n, 'p_true_best': float((win[0] == best).mean()),
                   'p_same_winner': float((win[0] == win[1]).mean()), 'gap_median': float(np.median(gaps))}
            if label:
                # con medias iguales no hay "mejor": la referencia es el azar (1/k)
                row['p_true_best'] = 1.0 / k
            for f, v in flags.items():
                row[f'p_{f}'] = float(v.mean()) if n >= 3 else np.nan
            rows.append({**{kk: vv for kk, vv in row.items() if kk in ('model', 'n_runs')},
                         **{kk + label: vv for kk, vv in row.items() if kk not in ('model', 'n_runs')}})
    out = pd.DataFrame(rows)
    return out.groupby(['model', 'n_runs'], as_index=False).first()


def architecture_pairs(df, n_boot):
    """Comparación directa de arquitecturas para cada optimizador, pareada por
    semilla (misma semilla => misma aleatoriedad del optimizador y del
    entrenamiento), con Holm dentro de cada optimizador."""
    core = df[(df.tag == 'core') & (df.budget == FULL)]
    rows = []
    for o, sub in core.groupby('optimizer', observed=True):
        wide = sub.pivot_table(index='seed', columns='model', values='all_mae').dropna()
        archs = [m for m in ('LSTM', 'GRU', 'MLP') if m in wide.columns]
        if len(wide) < 3 or len(archs) < 2:
            continue
        block = []
        for a, b in combinations(archs, 2):
            d = (wide[a] - wide[b]).to_numpy()
            try:
                p_w = stats.wilcoxon(d).pvalue
            except ValueError:
                p_w = 1.0
            lo95, hi95 = boot_ci(d, 0.95, n_boot)
            lo90, hi90 = boot_ci(d, 1 - 2 * C.TOST_ALPHA, n_boot)
            block.append({'optimizer': o, 'n_seeds': len(wide), 'A': a, 'B': b,
                          'mean_A': wide[a].mean(), 'mean_B': wide[b].mean(), 'mean_diff': d.mean(),
                          'ci95_lo': lo95, 'ci95_hi': hi95, 'ci90_lo': lo90, 'ci90_hi': hi90,
                          'p_wilcoxon': p_w, 'A12': a12(wide[a], wide[b]),
                          f'equiv_d{C.TOST_DELTA:g}': bool(-C.TOST_DELTA < lo90 and hi90 < C.TOST_DELTA)})
        for r, pa in zip(block, holm([r['p_wilcoxon'] for r in block])):
            r['p_holm'] = pa
            r['significant'] = bool(pa < C.ALPHA)
        rows += block
    return pd.DataFrame(rows)


def variance_decomposition(df):
    """ANOVA con bloque de semilla sobre el diseño balanceado (bloques completos)."""
    core = df[(df.tag == 'core') & (df.budget == FULL) & (df.optimizer != 'DEFAULT')]
    wide = core.pivot_table(index='seed', columns=['model', 'optimizer'], values='all_mae',
                            observed=True)
    cnt = wide.count()
    keep_opts = [o for o in wide.columns.get_level_values(1).unique()
                 if all(cnt.get((m, o), 0) >= max(3, 0.5 * cnt.max()) for m in C.MODELS)]
    wide = wide.loc[:, wide.columns.get_level_values(1).isin(keep_opts)].dropna()
    if len(wide) < 2:
        return pd.DataFrame()
    y = wide.to_numpy()                              # semillas x (modelo, opt)
    models = wide.columns.get_level_values(0)
    opts = wide.columns.get_level_values(1)
    grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    n_s, n_c = y.shape
    cell = y.mean(axis=0)
    m_mean = pd.Series(cell).groupby(models).transform('mean').to_numpy()
    o_mean = pd.Series(cell).groupby(opts).transform('mean').to_numpy()
    ss_model = n_s * ((m_mean - grand) ** 2).sum()
    ss_opt = n_s * ((o_mean - grand) ** 2).sum()
    ss_inter = n_s * ((cell - m_mean - o_mean + grand) ** 2).sum()
    ss_seed = n_c * ((y.mean(axis=1) - grand) ** 2).sum()
    ss_res = ss_tot - ss_model - ss_opt - ss_inter - ss_seed
    terms = {'architecture': ss_model, 'optimizer': ss_opt, 'architecture x optimizer': ss_inter,
             'seed (block)': ss_seed, 'residual': ss_res}
    return pd.DataFrame([{'term': k, 'SS': v, 'eta2': v / ss_tot, 'n_seeds': n_s,
                          'n_cells': n_c} for k, v in terms.items()])


def val_test_correlation(df):
    core = df[(df.tag == 'core') & (df.budget == FULL) & (df.optimizer != 'DEFAULT')]
    rows = []
    for m, sub in list(core.groupby('model')) + [('all', core)]:
        if len(sub) < 3:
            continue
        rows.append({'model': m, 'n': len(sub),
                     'pearson': stats.pearsonr(sub.best_val_mae, sub.all_mae)[0],
                     'spearman': stats.spearmanr(sub.best_val_mae, sub.all_mae)[0]})
    return pd.DataFrame(rows)


def low_budget_table(df):
    core = df[(df.tag == 'core') & (df.optimizer != 'DEFAULT')]
    return core.groupby(['model', 'optimizer', 'budget'], observed=True).agg(
        n=('seed', 'count'), mae_mean=('all_mae', 'mean'), mae_sd=('all_mae', 'std'),
        val_inc_mean=('val_mae_inc', 'mean')).reset_index()


def popsens_table(df):
    sub = df[(df.budget == FULL) & df.optimizer.isin(C.POP_SENS_OPTIMIZERS)
             & df.model.isin(C.POP_SENS_MODELS)]
    sub = sub[(sub.tag == 'popsens') | (sub.pop == C.POP_SIZE)]
    return sub.groupby(['model', 'optimizer', 'pop'], observed=True).agg(
        n=('seed', 'count'), mae_mean=('all_mae', 'mean'), mae_sd=('all_mae', 'std'),
        val_mean=('best_val_mae', 'mean')).reset_index()


def anytime_curves(hist):
    rows = []
    for (tag, run), h in hist.items():
        if tag != 'core' or not h:
            continue
        model, opt = run.split('_')[0], run.split('_')[1]
        for e in h:
            rows.append({'model': model, 'optimizer': opt, 'run': run, 'eval': e['eval'],
                         'cum_epochs': e['cum_epochs'], 'best_so_far': e['best_so_far']})
    a = pd.DataFrame(rows)
    if a.empty:
        return a, a
    by_eval = a.groupby(['model', 'optimizer', 'eval']).best_so_far.agg(
        ['mean', 'std', 'count']).reset_index()
    a['epoch_bin'] = (np.ceil(a.cum_epochs / 50) * 50).astype(int)
    by_ep = (a.groupby(['model', 'optimizer', 'run', 'epoch_bin']).best_so_far.min()
             .groupby(['model', 'optimizer', 'epoch_bin']).agg(['mean', 'std', 'count'])
             .reset_index())
    return by_eval, by_ep


def hp_importance(hist, n_repeats=30):
    """Permutación sobre un random forest ajustado a las evaluaciones de RS."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    rows = []
    for model in C.MODELS:
        X, y = [], []
        for (tag, run), h in hist.items():
            if tag == 'core' and run.startswith(f'{model}_RS_'):
                for e in h:
                    if e['val_mae'] < 9000:
                        p = e['params']
                        X.append([p['units_1'], p['units_2'], p['dropout_rate'],
                                  np.log10(p['learning_rate']), p['batch_size']])
                        y.append(e['val_mae'])
        if len(y) < 50:
            continue
        X, y = np.array(X), np.array(y)
        rf = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1).fit(X, y)
        imp = permutation_importance(rf, X, y, n_repeats=n_repeats, random_state=0, n_jobs=-1)
        tot = np.clip(imp.importances_mean, 0, None).sum()
        for name, mu, sd in zip(['units_1', 'units_2', 'dropout_rate', 'log10_lr', 'batch_size'],
                                imp.importances_mean, imp.importances_std):
            rows.append({'model': model, 'n_evals': len(y), 'hyperparameter': name,
                         'importance': mu, 'importance_sd': sd,
                         'relative': max(mu, 0) / tot if tot > 0 else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=10000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    df, hist = load_runs()
    df.to_csv(os.path.join(OUT, 'runs_long.csv'), index=False)
    refs = references()
    refs.to_csv(os.path.join(OUT, 'reference_forecasts.csv'), index=False)

    tables = {'summary_B100': summary_table(df), 'hpo_gain_vs_default': hpo_gain(df),
              'variance_decomposition': variance_decomposition(df),
              'architecture_pairwise': architecture_pairs(df, args.boot),
              'val_test_correlation': val_test_correlation(df),
              'low_budget': low_budget_table(df), 'popsens': popsens_table(df)}
    fr_all, pr_all = [], []
    for metric in ('all_mae', 'day_ghi50_mae', 'roll_mae'):
        for b in (BUDGETS if metric == 'all_mae' else [FULL]):
            fr, pr = friedman_and_pairs(df, metric, b, args.boot)
            fr_all.append(fr)
            pr_all.append(pr)
    tables['friedman'] = pd.concat(fr_all, ignore_index=True)
    tables['pairwise'] = pd.concat(pr_all, ignore_index=True)
    tables['anytime_by_eval'], tables['anytime_by_epoch'] = anytime_curves(hist)
    tables['hp_importance_rs'] = hp_importance(hist)
    tables['single_run_pairs'], tables['single_run_overall'] = single_run_reliability(df)
    tables['literature_designs'] = literature_designs(df)

    for name, t in tables.items():
        t.to_csv(os.path.join(OUT, f'{name}.csv'), index=False)

    # Resumen legible en consola
    n_runs = df[df.budget == FULL].groupby('tag').size().to_dict()
    print(f'Corridas: {n_runs}')
    ref = refs[(refs.split == 'test') & (refs.block == 'all')].set_index('reference')
    print('Referencias test (todas las horas): ' + ', '.join(
        f'{k} MAE={v.mae:.2f} RMSE={v.rmse:.2f}' for k, v in ref.iterrows()))
    s = tables['summary_B100']
    print(s[['model', 'optimizer', 'n', 'mae_mean', 'mae_sd', 'day_mae_mean', 'skill_smart_mean',
             'roll_mae_mean']].round(3).to_string(index=False))
    pr = tables['pairwise']
    if not pr.empty:
        main_pr = pr[(pr.metric == 'all_mae') & (pr.budget == FULL)]
        for m, g in main_pr.groupby('model'):
            print(f'{m}: n={g.n_seeds.iloc[0]} pares={len(g)} significativos(Holm)={g.significant.sum()} '
                  f'equivalentes δ=1: {g[f"equiv_d{C.TOST_DELTA:g}"].sum()} '
                  f'MDE80 mediana={g.mde_80.median():.2f}')
    print(tables['variance_decomposition'].round(4).to_string(index=False))
    so = tables['single_run_overall']
    if not so.empty:
        print('Estudio de una sola corrida:')
        print(so[['model', 'k', 'best_by_mean', 'p_single_run_picks_best', 'mean_spread_30seed',
                  'median_spread_single_run', 'p_spread_gt_delta']].round(3).to_string(index=False))
        sp = tables['single_run_pairs']
        print(sp.groupby('model')[['p_gap_gt_delta', 'p_reversal', 'n_req_tost']]
              .median().round(3).to_string())
    print(f'Tablas en {OUT}')


if __name__ == '__main__':
    main()
