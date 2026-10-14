"""Experimento de mayor fidelidad: MLP con hasta 50 épocas por evaluación.

Compara RS, TPE y PSO (10 semillas, B = 100) entrenados con el objetivo de
alta fidelidad (outputs/results/hifi/) con las mismas semillas del experimento
principal (objetivo de 10 épocas, outputs/results/core/):
  - correlación entre el MAE de validación de la HPO y el MAE de test;
  - ventaja de los métodos guiados sobre RS en validación y en test;
  - pruebas pareadas (Wilcoxon + Holm) y equivalencia (IC BCa 90 %, delta = 1 W/m²).

Uso:  python fidelity_check.py
"""
import glob
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import config as C

OPTS = ['RS', 'TPE', 'PSO']


def load(tag):
    rows = []
    for f in glob.glob(os.path.join(C.RES_DIR, tag, 'MLP_*.json')):
        r = json.load(open(f))
        if r['optimizer'] not in OPTS or r['seed'] > 9:
            continue
        F = r['finals']['B100']
        while 'same_as' in F:
            F = r['finals'][F['same_as']]
        rows.append({'tag': tag, 'opt': r['optimizer'], 'seed': r['seed'], 'val_hpo': r['best_val_mae'],
                     'val_final': F['val_mae_final'], 'test': F['test']['all']['mae'],
                     'epochs': r['total_hpo_epochs'], 'hours': r['hpo_time_s'] / 3600})
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


D = pd.concat([load('core'), load('hifi')])
seeds = sorted(set(D[D.tag == 'hifi'].seed))
D = D[D.seed.isin(seeds)]
out = {'n_seeds': len(seeds)}
for tag, g in D.groupby('tag'):
    res = {'mean_test': g.groupby('opt').test.mean().to_dict(), 'mean_val_hpo': g.groupby('opt').val_hpo.mean().to_dict(),
           'rho_valhpo_test': float(stats.spearmanr(g.val_hpo, g.test)[0]),
           'rho_valhpo_valfinal': float(stats.spearmanr(g.val_hpo, g.val_final)[0]),
           'epochs': g.groupby('opt').epochs.mean().to_dict(), 'hours': g.groupby('opt').hours.mean().to_dict()}
    W = g.pivot(index='seed', columns='opt', values='test')[OPTS]
    V = g.pivot(index='seed', columns='opt', values='val_hpo')[OPTS]
    pv, pairs = [], []
    for a, b in combinations(OPTS, 2):
        d = (W[a] - W[b]).to_numpy()
        pv.append(stats.wilcoxon(d).pvalue)
        ci = stats.bootstrap((d,), np.mean, confidence_level=0.90, n_resamples=10000, method='BCa', random_state=0).confidence_interval
        pairs.append({'pair': f'{a}-{b}', 'mean_diff': float(d.mean()), 'ci90': [float(ci.low), float(ci.high)],
                      'equiv_d1': bool(-1 < ci.low and ci.high < 1),
                      'val_diff': float((V[a] - V[b]).mean())})
    for pr, q in zip(pairs, holm(pv)):
        pr['p_holm'] = float(q)
    res['pairs'] = pairs
    out[tag] = res
# ¿la mayor fidelidad mejora el modelo final? (mismas semillas, pareado)
for o in OPTS:
    a = D[(D.tag == 'hifi') & (D.opt == o)].set_index('seed').test
    b = D[(D.tag == 'core') & (D.opt == o)].set_index('seed').test
    d = (a - b).dropna()
    out.setdefault('hifi_minus_core', {})[o] = {'mean': float(d.mean()), 'p': float(stats.wilcoxon(d).pvalue)}
json.dump(out, open(os.path.join(C.OUT_DIR, 'analysis', 'fidelity_check.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
