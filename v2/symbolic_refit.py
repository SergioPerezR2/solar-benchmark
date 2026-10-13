"""
Reajuste de la ecuación simbólica (Ec. 9 del manuscrito) sobre el split v2 y
bootstrap por bloques de días para los intervalos de sus 6 parámetros.

    python symbolic_refit.py [--boot 300]

La forma funcional y la geometría solar son exactamente las del manuscrito;
solo cambian los datos de ajuste (train 2019–2020, hora local UTC−6 fija).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

import config as C

EPS = 1e-3
BOUNDS = [(0, 2000), (0, 2), (-2, 2), (-2, 2), (-1, 1), (-np.pi, np.pi)]


def load():
    df = pd.read_csv(C.RAW_CSV, index_col='TIMESTAMP', parse_dates=True).sort_index()
    local = df.index.tz_convert(None) + pd.Timedelta(hours=C.UTC_OFFSET_H)
    df['h'], df['d'], df['year'] = local.hour, local.dayofyear, local.year
    phi = np.radians(C.LATITUDE)
    dec = np.radians(23.45) * np.sin(2 * np.pi * (df['d'] - 81) / 365)
    omega = np.radians(15 * (df['h'] - 12))
    df['cosz'] = np.maximum(0, np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(omega))
    tr = df['year'].isin(C.SPLIT_YEARS['train'])
    for c, n in (('TempAir', 'T_s'), ('WindSpeed', 'v_s'), ('d', 'd_s')):
        lo, hi = df.loc[tr, c].min(), df.loc[tr, c].max()
        df[n] = (df[c] - lo) / (hi - lo)
    return df


def predict(p, X):
    cz, T, v, d = X
    return (p[0] * cz * np.exp(-p[1] / (cz + EPS))
            * np.maximum(0, 1 + p[2] * T + p[3] * v)
            * np.maximum(0, 1 + p[4] * np.sin(2 * np.pi * d + p[5])))


def fit(X, y, p0=None, seed=0):
    if p0 is None:
        p0 = differential_evolution(lambda p: np.mean((predict(p, X) - y) ** 2), BOUNDS,
                                    seed=seed, tol=1e-8, maxiter=2000, polish=True).x
    lo, hi = np.array(BOUNDS).T
    return least_squares(lambda p: predict(p, X) - y, np.clip(p0, lo, hi), bounds=(lo, hi)).x


def metrics(y, p, mask):
    e = p[mask] - y[mask]
    return {'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt((e ** 2).mean())),
            'mbe': float(e.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=300)
    args = ap.parse_args()
    df = load()
    cols = ['cosz', 'T_s', 'v_s', 'd_s']
    tr = df['year'].isin(C.SPLIT_YEARS['train']).to_numpy()
    te = df['year'].isin(C.SPLIT_YEARS['test']).to_numpy()
    X = df[cols].to_numpy().T
    y = df['GHI'].to_numpy()

    p_hat = fit(X[:, tr], y[tr])
    pred = predict(p_hat, X)
    day = y > C.DAY_THRESHOLD_GHI
    res = {'params': p_hat.tolist(),
           'test_all': metrics(y, pred, te), 'test_day_ghi50': metrics(y, pred, te & day),
           'train_all': metrics(y, pred, tr)}

    # Bootstrap por bloques de días completos (preserva la autocorrelación intradía)
    rng = np.random.default_rng(0)
    days = df.loc[tr].index.floor('D').to_numpy()
    uniq = np.unique(days)
    idx_by_day = {d: np.where(days == d)[0] for d in uniq}
    Xtr, ytr = X[:, tr], y[tr]
    boots = []
    for _ in range(args.boot):
        sel = np.concatenate([idx_by_day[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        boots.append(fit(Xtr[:, sel], ytr[sel], p0=p_hat))
    boots = np.array(boots)
    res['bootstrap'] = {'n': args.boot, 'ci95_lo': np.percentile(boots, 2.5, axis=0).tolist(),
                        'ci95_hi': np.percentile(boots, 97.5, axis=0).tolist(),
                        'sd': boots.std(axis=0, ddof=1).tolist()}

    out = os.path.join(C.OUT_DIR, 'analysis')
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, 'symbolic_refit.json'), 'w'), indent=2)
    names = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']
    for n, v, lo, hi in zip(names, p_hat, res['bootstrap']['ci95_lo'], res['bootstrap']['ci95_hi']):
        print(f'{n} = {v:9.4f}   IC95 [{lo:9.4f}, {hi:9.4f}]')
    print('test todas las horas:', {k: round(v, 2) for k, v in res['test_all'].items()})
    print('test diurno (GHI>50):', {k: round(v, 2) for k, v in res['test_day_ghi50'].items()})


if __name__ == '__main__':
    main()
