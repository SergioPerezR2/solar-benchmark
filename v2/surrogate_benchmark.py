"""
Benchmark sustituto: permite evaluar un optimizador nuevo sin reentrenar redes.

Con todas las evaluaciones completas registradas en el experimento (configuración
u ∈ [0,1]^5 y MAE de validación) se ajusta, para cada arquitectura, un modelo de
regresión que predice el MAE de validación de cualquier configuración; el ruido
entre semillas se emula con el error residual fuera de muestra. Luego:

  1. Fidelidad: validación cruzada agrupada por semilla (el modelo nunca ve las
     corridas de la semilla con la que se evalúa).
  2. Réplica: se ejecutan los optimizadores con el MISMO código de hpo.py sobre el
     sustituto (30 semillas, B = 100) y se comparan sus curvas y su orden con los
     del experimento real.

    python surrogate_benchmark.py              # ajuste, fidelidad y réplica
    python surrogate_benchmark.py --no-replay  # solo ajuste y fidelidad

Uso del sustituto con un optimizador propio (ver README):

    from surrogate_benchmark import SurrogateObjective
    f = SurrogateObjective('LSTM', seed=0)       # f(u) -> MAE de validación (W/m²)

Salidas en outputs/surrogate/: <MODEL>.joblib, fidelity.csv, replay_*.csv.
BOHB no se incluye en la réplica porque el sustituto no modela curvas de aprendizaje.
"""
import argparse
import glob
import json
import os
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

import config as C
import hpo

OUT = os.path.join(C.OUT_DIR, 'surrogate')
OPTS = ['PSO', 'GA', 'ABC', 'CMAES', 'TPE', 'RS']
FAIL = 9000.0                       # evaluaciones fallidas se registran con un MAE muy grande


def load_evaluations():
    rows = []
    for path in glob.glob(os.path.join(C.RES_DIR, 'core', '*.json')):
        r = json.load(open(path))
        if r.get('dry') or r['optimizer'] == 'DEFAULT':
            continue
        for e in r['history']:
            if e.get('pruned') or e['val_mae'] >= FAIL:
                continue
            rows.append({'model': r['model'], 'optimizer': r['optimizer'], 'seed': r['seed'],
                         **{f'u{i}': v for i, v in enumerate(e['u'])}, 'val_mae': e['val_mae']})
    return pd.DataFrame(rows)


def make_regressor(seed=0):
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=20, l2_regularization=1.0,
                                         random_state=seed)


U = [f'u{i}' for i in range(C.DIM)]


def fit_and_validate(ev, n_folds=5):
    """Ajusta un modelo por arquitectura y mide su fidelidad con folds por semilla."""
    os.makedirs(OUT, exist_ok=True)
    fid = []
    for m, g in ev.groupby('model'):
        folds = np.array(g.seed) % n_folds
        pred = np.empty(len(g))
        for f in range(n_folds):
            tr, te = folds != f, folds == f
            reg = make_regressor().fit(g.loc[tr, U], np.log(g.val_mae[tr]))
            pred[te] = np.exp(reg.predict(g.loc[te, U]))
        y = g.val_mae.to_numpy()
        resid = np.log(y) - np.log(pred)
        # El residuo se separa en un desplazamiento común a cada corrida (misma semilla
        # y optimizador) y en ruido de cada evaluación dentro de la corrida.
        run = g.optimizer.astype(str) + '_' + g.seed.astype(str)
        run_mean = pd.Series(resid, index=g.index).groupby(run.values).transform('mean').to_numpy()
        sigma_w = float((resid - run_mean).std(ddof=1))
        sigma_b = float(pd.Series(resid).groupby(run.values).mean().std(ddof=1))
        reg = make_regressor().fit(g[U], np.log(g.val_mae))
        joblib.dump({'model': reg, 'sigma_w': sigma_w, 'sigma_b': sigma_b, 'n': len(g)},
                    os.path.join(OUT, f'{m}.joblib'))
        # ranking de configuraciones buenas: ¿el sustituto distingue el 10 % mejor?
        top = y <= np.quantile(y, 0.10)
        fid.append({'model': m, 'n_evals': len(g),
                    'r2': 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                    'spearman': stats.spearmanr(y, pred)[0],
                    'mae_pred': float(np.abs(y - pred).mean()),
                    'spearman_top10': stats.spearmanr(y[top], pred[top])[0],
                    'sigma_log': float(resid.std(ddof=1)), 'sigma_within': sigma_w, 'sigma_run': sigma_b})
    fid = pd.DataFrame(fid)
    fid.to_csv(os.path.join(OUT, 'fidelity.csv'), index=False)
    return fid


class SurrogateObjective:
    """Función objetivo sustituta: u ∈ [0,1]^5 -> MAE de validación (W/m²).

    noise=True emula la variabilidad del entrenamiento estimada fuera de muestra:
    un desplazamiento común a toda la corrida (fijado por `seed`) y ruido
    independiente en cada evaluación.
    """

    def __init__(self, model, seed=0, noise=True):
        d = joblib.load(os.path.join(OUT, f'{model}.joblib'))
        self.reg, self.noise = d['model'], noise
        self.sigma_w = d['sigma_w'] * d.get('kappa', 1.0)
        self.rng = np.random.default_rng(seed)
        self.offset = float(self.rng.normal(0, d['sigma_b'])) if noise else 0.0

    def __call__(self, u):
        mu = float(np.exp(self.reg.predict(pd.DataFrame([np.clip(u, 0, 1)], columns=U))[0]))
        if not self.noise:
            return mu
        return mu * float(np.exp(self.offset + self.rng.normal(0, self.sigma_w)))


def calibrate(n_seeds=200):
    """Escala el ruido por evaluación (kappa) para que Random Search sobre el sustituto
    alcance, tras B evaluaciones, el mismo MAE de validación medio que en el experimento.
    Solo se usa RS, que no explota la estructura del sustituto; los otros cinco
    métodos quedan como validación independiente."""
    real = {}
    for path in glob.glob(os.path.join(C.RES_DIR, 'core', '*_RS_s*.json')):
        r = json.load(open(path))
        real.setdefault(r['model'], []).append(r['history'][C.BUDGET - 1]['best_so_far'])
    out = {}
    for m in C.MODELS:
        path = os.path.join(OUT, f'{m}.joblib')
        d = joblib.load(path)
        rng = np.random.default_rng(123)
        u = rng.random((n_seeds, C.BUDGET, C.DIM))
        mu = np.exp(d['model'].predict(pd.DataFrame(u.reshape(-1, C.DIM), columns=U))).reshape(n_seeds, C.BUDGET)
        off = rng.normal(0, d['sigma_b'], (n_seeds, 1))
        z = rng.normal(0, 1, (n_seeds, C.BUDGET))
        target = float(np.mean(real[m]))
        grid = np.linspace(0.0, 1.0, 101)
        err = [abs(float((mu * np.exp(off + k * d['sigma_w'] * z)).min(axis=1).mean()) - target) for k in grid]
        d['kappa'] = float(grid[int(np.argmin(err))])
        joblib.dump(d, path)
        out[m] = {'kappa': d['kappa'], 'rs_real_b100': target}
    return out


def replay(seeds=range(30)):
    """Ejecuta los optimizadores de hpo.py sobre el sustituto y guarda sus historias."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows, times = [], []
    for m in C.MODELS:
        for o in OPTS:
            for s in seeds:
                f = SurrogateObjective(m, seed=10_000 + s)
                best, hist = np.inf, []

                def fn(u, report=None, f=f):
                    return f(u), C.OPT_EPOCHS, False

                ev = hpo.Evaluator(fn, C.BUDGET)
                orig = ev.evaluate

                def logged(u, report=None):
                    nonlocal best
                    v = orig(u, report)
                    best = min(best, v)
                    hist.append(best)
                    return v

                ev.evaluate = logged
                t0 = time.time()
                hpo.OPTIMIZERS[o](ev, s, pop=C.POP_SIZE if o in ('PSO', 'GA', 'ABC') else None)
                times.append(time.time() - t0)
                for b in (10, 25, 50, 100):
                    rows.append({'model': m, 'optimizer': o, 'seed': s, 'eval': b, 'best_so_far': hist[b - 1]})
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(OUT, 'replay_runs.csv'), index=False)
    return rep, float(np.mean(times))


def compare(rep):
    """Compara la réplica con el experimento real (mejor MAE de validación hasta b)."""
    real = []
    for path in glob.glob(os.path.join(C.RES_DIR, 'core', '*.json')):
        r = json.load(open(path))
        if r.get('dry') or r['optimizer'] not in OPTS:
            continue
        h = r['history']
        for b in (10, 25, 50, 100):
            real.append({'model': r['model'], 'optimizer': r['optimizer'], 'seed': r['seed'],
                         'eval': b, 'best_so_far': h[b - 1]['best_so_far']})
    real = pd.DataFrame(real)
    a = real.groupby(['model', 'optimizer', 'eval']).best_so_far.mean().rename('real')
    b = rep.groupby(['model', 'optimizer', 'eval']).best_so_far.mean().rename('surrogate')
    cmp_ = pd.concat([a, b], axis=1).reset_index()
    cmp_.to_csv(os.path.join(OUT, 'replay_vs_real.csv'), index=False)
    summ = []
    for (m, e), g in cmp_.groupby(['model', 'eval']):
        summ.append({'model': m, 'eval': e, 'spearman_rank': stats.spearmanr(g.real, g.surrogate)[0],
                     'mean_abs_diff': float((g.real - g.surrogate).abs().mean()),
                     'real_best': g.loc[g.real.idxmin(), 'optimizer'],
                     'surrogate_best': g.loc[g.surrogate.idxmin(), 'optimizer'],
                     'real_worst': g.loc[g.real.idxmax(), 'optimizer'],
                     'surrogate_worst': g.loc[g.surrogate.idxmax(), 'optimizer']})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, 'replay_summary.csv'), index=False)
    allr = stats.spearmanr(cmp_.real, cmp_.surrogate)[0]
    return cmp_, summ, allr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-replay', action='store_true')
    args = ap.parse_args()
    ev = load_evaluations()
    print(f'{len(ev):,} evaluaciones completas')
    print(fit_and_validate(ev).round(3).to_string(index=False))
    if args.no_replay:
        return
    print('calibración con RS:', calibrate())
    rep, t = replay()
    cmp_, summ, allr = compare(rep)
    print(summ.round(3).to_string(index=False))
    print(f'Spearman global (18 celdas x 4 presupuestos): {allr:.3f}; segundos por corrida sustituta: {t:.2f}')
    json.dump({'spearman_all': allr, 'seconds_per_run': t}, open(os.path.join(OUT, 'replay_meta.json'), 'w'))


if __name__ == '__main__':
    main()
