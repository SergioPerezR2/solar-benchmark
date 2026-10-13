"""Análisis de robustez del benchmark v2.

Complementa analyze.py con comprobaciones que no cambian el diseño ni el
endpoint primario:

  1. Endpoints alternativos: MAE de todas las horas con recorte físico
     (predicción = 0 con el sol bajo el horizonte y no negatividad), MAE diurno
     por elevación solar (> 5°) y por GHI observado (> 50 W/m²).
  2. Descomposición de varianza con F, p y omega² (corrige por grados de libertad).
  3. Validación-test: separa el cambio de fidelidad (10 → 100 épocas) del
     cambio de año usando el MAE de validación del modelo final reentrenado.
  4. Diseño inicial compartido entre PSO, GA, ABC y RS para una misma semilla.
  5. Valores analíticos de referencia para la emulación de diseños de la
     literatura (la emulación está en analyze.py).
  6. Frecuencia de la fase de exploradoras (scouts) en ABC.

Uso:  python robustness.py [salida]      (por defecto outputs/analysis)
"""
import glob
import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import config as C
import hpo

RES = os.path.join(C.RES_DIR, 'core')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join('outputs', 'analysis')
OPTS = ['PSO', 'GA', 'ABC', 'CMAES', 'TPE', 'BOHB', 'RS']
MODELS = ['MLP', 'LSTM', 'GRU']
ALPHA, DELTA = 0.05, 1.0


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def ci90(d, n_boot=10000):
    # mismo criterio que analyze.py: intervalo BCa del 90 % de la diferencia media pareada
    r = stats.bootstrap((d,), np.mean, confidence_level=0.90, n_resamples=n_boot,
                        method='BCa', random_state=0)
    return r.confidence_interval.low, r.confidence_interval.high


# ── Carga ─────────────────────────────────────────────────────────────────────
ds = np.load(os.path.join(C.DATA_DIR, 'dataset.npz'))
test = ds['split_test']
y = ds['ghi'][test].astype(float)
elev = ds['elevation'][test].astype(float)
smart = ds['smart_persistence'][test].astype(float)
masks = {'all': np.ones_like(y, bool), 'all_clipped': np.ones_like(y, bool),
         'day_elev5': ds['day_elev'][test].astype(bool), 'day_ghi50': ds['day_ghi'][test].astype(bool)}

recs, rows = {}, []
for f in sorted(glob.glob(os.path.join(RES, '*.json'))):
    r = json.load(open(f))
    if r['optimizer'] == 'DEFAULT':
        continue
    pred = np.load(f[:-5] + '.npz')['test_B100'].astype(float)
    clip = np.where(elev <= 0, 0.0, np.maximum(pred, 0.0))
    row = {'model': r['model'], 'opt': r['optimizer'], 'seed': r['seed']}
    for e, m in masks.items():
        p = clip if e == 'all_clipped' else pred
        row[e] = float(np.mean(np.abs(p[m] - y[m])))
    night = elev <= 0
    row['night_abs_pred'] = float(np.mean(np.abs(pred[night])))
    row['night_contrib'] = float(np.sum(np.abs(pred[~masks['day_ghi50']] - y[~masks['day_ghi50']])) / len(y))
    F = r['finals']['B100']
    row.update(val_hpo=F['val_mae_hpo'], val_final=F['val_mae_final'], test_check=F['test']['all']['mae'],
               hpo_epochs=r['total_hpo_epochs'], n_evals=r['n_evals'])
    rows.append(row)
    recs[(r['model'], r['optimizer'], r['seed'])] = r
D = pd.DataFrame(rows)
assert np.allclose(D['all'], D['test_check'], atol=1e-6), 'las predicciones no reproducen el MAE guardado'
num = {}

# ── 1. Endpoints alternativos ─────────────────────────────────────────────────
ref_mae = {e: float(np.mean(np.abs(smart[m] - y[m]))) for e, m in masks.items()}
ref_mae['all_clipped'] = ref_mae['all']           # smart persistence ya es 0 de noche
ep_rows = []
for e in masks:
    for m in MODELS:
        W = D[D.model == m].pivot(index='seed', columns='opt', values=e)[OPTS]
        pv, eq1, eq05, diffs = [], 0, 0, []
        for a, b in combinations(OPTS, 2):
            d = (W[a] - W[b]).to_numpy()
            pv.append(stats.wilcoxon(d).pvalue)
            lo, hi = ci90(d)
            eq1 += -1.0 < lo and hi < 1.0
            eq05 += -0.5 < lo and hi < 0.5
            diffs.append(abs(d.mean()))
        mu = W.mean()
        ep_rows.append({'endpoint': e, 'model': m, 'mae_min': mu.min(), 'mae_max': mu.max(),
                        'smart_mae': ref_mae[e], 'skill_mae_min': 1 - mu.max() / ref_mae[e],
                        'skill_mae_max': 1 - mu.min() / ref_mae[e],
                        'gain_min': ref_mae[e] - mu.max(), 'gain_max': ref_mae[e] - mu.min(),
                        'n_sig_holm': int((holm(pv) < ALPHA).sum()), 'n_equiv_d1': eq1, 'n_equiv_d05': eq05,
                        'max_abs_diff': max(diffs)})
EP = pd.DataFrame(ep_rows)
EP.to_csv(os.path.join(OUT, 'robust_endpoints.csv'), index=False)
for e, g in EP.groupby('endpoint'):
    num[f'rob_{e}_equiv_d1'] = int(g.n_equiv_d1.sum())
    num[f'rob_{e}_equiv_d05'] = int(g.n_equiv_d05.sum())
    num[f'rob_{e}_sig'] = int(g.n_sig_holm.sum())
    num[f'rob_{e}_maxdiff'] = float(g.max_abs_diff.max())
for _, r in EP.iterrows():
    k = f"rob_{r.endpoint}_{r.model}"
    num[k + '_skill'] = [float(r.skill_mae_min), float(r.skill_mae_max)]
    num[k + '_gain'] = [float(r.gain_min), float(r.gain_max)]
    num[k + '_smart'] = float(r.smart_mae)
num['night_abs_pred'] = D.groupby('model').night_abs_pred.mean().to_dict()
num['night_contrib'] = D.groupby('model').night_contrib.mean().to_dict()
num['night_contrib_smart'] = float(np.sum(np.abs(smart[~masks['day_ghi50']] - y[~masks['day_ghi50']])) / len(y))
num['clip_gain'] = (D['all'] - D['all_clipped']).groupby(D.model).mean().to_dict()

# ── 2. Varianza con omega² ────────────────────────────────────────────────────
V = pd.read_csv(os.path.join(OUT, 'variance_decomposition.csv')).set_index('term')
df_ = {'architecture': 2, 'optimizer': 6, 'architecture x optimizer': 12, 'seed (block)': 29}
ss_tot = V.SS.sum()
df_res = 3 * 7 * 30 - 1 - sum(df_.values())
ms_res = V.loc['residual', 'SS'] / df_res
vrows = []
for t, k in df_.items():
    ss = V.loc[t, 'SS']
    Fv = (ss / k) / ms_res
    vrows.append({'term': t, 'df': k, 'SS': ss, 'F': Fv, 'p': float(stats.f.sf(Fv, k, df_res)),
                  'eta2': ss / ss_tot, 'omega2': max(0.0, (ss - k * ms_res) / (ss_tot + ms_res))})
vrows.append({'term': 'residual', 'df': df_res, 'SS': V.loc['residual', 'SS'], 'F': np.nan, 'p': np.nan,
              'eta2': V.loc['residual', 'SS'] / ss_tot, 'omega2': np.nan})
VV = pd.DataFrame(vrows)
VV.to_csv(os.path.join(OUT, 'variance_omega2.csv'), index=False)
for _, r in VV.iterrows():
    key = {'architecture': 'arch', 'optimizer': 'opt', 'architecture x optimizer': 'axo',
           'seed (block)': 'seed', 'residual': 'res'}[r.term]
    num[f'var_{key}_df'] = int(r.df)
    num[f'var_{key}_omega2'] = None if pd.isna(r.omega2) else float(r.omega2)
    num[f'var_{key}_F'] = None if pd.isna(r.F) else float(r.F)
    num[f'var_{key}_p'] = None if pd.isna(r.p) else float(r.p)

# ── 3. Fidelidad frente a año ─────────────────────────────────────────────────
for m, g in D.groupby('model'):
    num[f'vt_{m}_hpo_test'] = float(stats.spearmanr(g.val_hpo, g['all'])[0])
    num[f'vt_{m}_final_test'] = float(stats.spearmanr(g.val_final, g['all'])[0])
    num[f'vt_{m}_hpo_final'] = float(stats.spearmanr(g.val_hpo, g.val_final)[0])
    num[f'vt_{m}_optimism'] = float((g.val_final - g.val_hpo).mean())

# ── 4. Diseño inicial compartido ──────────────────────────────────────────────
shared = {'PSO_GA': 0, 'PSO_RS': 0, 'PSO_ABC5': 0, 'n': 0}
for m in MODELS:
    for s in range(30):
        u = {o: np.array([h['u'] for h in recs[(m, o, s)]['history'][:10]]) for o in ('PSO', 'GA', 'RS', 'ABC')}
        shared['n'] += 1
        shared['PSO_GA'] += np.allclose(u['PSO'], u['GA'])
        shared['PSO_RS'] += np.allclose(u['PSO'], u['RS'])
        shared['PSO_ABC5'] += np.allclose(u['PSO'][:5], u['ABC'][:5])
num['shared_init'] = shared
icorr = []
for m in MODELS:
    W = D[D.model == m].pivot(index='seed', columns='opt', values='all')[OPTS]
    Cm = W.corr(method='spearman').to_numpy()
    icorr.append(Cm[np.triu_indices(7, 1)].mean())
num['intra_seed_corr'] = dict(zip(MODELS, map(float, icorr)))

# ── 5. Valores analíticos de referencia para la emulación ────────────────────
# (la emulación de diseños de la literatura, con su nulo, está en analyze.py)
num['fwer_6_uncorrected'] = 1 - (1 - ALPHA) ** 6
num['fwer_21_uncorrected'] = 1 - (1 - ALPHA) ** 21
num['mw_min_p_3v3'] = float(stats.mannwhitneyu([0, 1, 2], [3, 4, 5], alternative='two-sided').pvalue)


# ── 6. ABC: frecuencia de exploradoras ────────────────────────────────────────
class Replay:
    def __init__(self, hist):
        self.h, self.i, self.us = hist, 0, []

    def evaluate(self, u, report=None):
        if self.i >= len(self.h):
            raise hpo.BudgetExhausted()
        self.us.append(np.array(u, float))   # copia: ABC modifica S en sitio
        v = self.h[self.i]['val_mae']
        self.i += 1
        return v


scouts, match = [], 0
orig_random = None
for (m, o, s), r in recs.items():
    if o != 'ABC':
        continue
    ev = Replay(r['history'])
    count = {'n': 0}
    rng_cls = np.random.default_rng

    class Counting:
        """Cuenta las llamadas rng.random(DIM) fuera de la inicialización (scouts)."""
        def __init__(self, seed):
            self.g = rng_cls(seed)
            self.started = False

        def random(self, size=None):
            if self.started and size == C.DIM:
                count['n'] += 1
            self.started = True
            return self.g.random(size)

        def __getattr__(self, a):
            return getattr(self.g, a)

    hpo.np.random.default_rng = Counting
    try:
        hpo.run_abc(ev, s)
    finally:
        hpo.np.random.default_rng = rng_cls
    match += np.allclose(np.array(ev.us), np.array([h['u'] for h in r['history']]))
    scouts.append(count['n'])
num['abc_replay_match'] = int(match)
num['abc_runs'] = len(scouts)
num['abc_scouts_mean'] = float(np.mean(scouts))
num['abc_runs_with_scout'] = int(np.sum(np.array(scouts) > 0))

# ── 7. Otros datos de control ─────────────────────────────────────────────────
b = D[D.opt == 'BOHB']
num['bohb_epochs'] = [float(b.hpo_epochs.min()), float(b.hpo_epochs.max())]
num['n_failed_evals'] = int(sum(h['val_mae'] >= 9999 for r in recs.values() for h in r['history']))

json.dump(num, open(os.path.join(OUT, 'robustness.json'), 'w'), indent=1, default=float)
print(EP.round(3).to_string())
print(VV.round(4).to_string())
print(json.dumps({k: v for k, v in num.items() if not k.startswith('rob_')}, indent=1, default=float))
