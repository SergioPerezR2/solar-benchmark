"""
Preprocesamiento v2: genera el tensor de ventanas, los índices de cada split
(principal y rolling-origin), las máscaras diurnas y los baselines de
persistencia. Se ejecuta una sola vez antes del benchmark.

    python prepare_data.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import pvlib
from sklearn.preprocessing import MinMaxScaler

import config as C


def load_raw():
    df = pd.read_csv(C.RAW_CSV, index_col='TIMESTAMP', parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df.sort_index()

    n_raw = len(df)
    n_dup = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep='first')]

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='1h')
    n_missing = len(full_idx.difference(df.index))
    df = df.reindex(full_idx)

    # Filtros físicos; los valores inválidos se tratan como huecos
    bad = (df['GHI'] < 0) | ~df['TempAir'].between(-20, 60) | ~df['WindSpeed'].between(0, 50)
    n_bad = int(bad.sum())
    df.loc[bad, ['GHI', 'TempAir', 'WindSpeed']] = np.nan
    n_nan = int(df[['GHI', 'TempAir', 'WindSpeed']].isna().any(axis=1).sum())
    df[['GHI', 'TempAir', 'WindSpeed']] = df[['GHI', 'TempAir', 'WindSpeed']].interpolate(
        method='time', limit=3)
    n_left = int(df[['GHI', 'TempAir', 'WindSpeed']].isna().any(axis=1).sum())
    if n_left:
        raise RuntimeError(f'{n_left} huecos > 3 h; revisar antes de continuar')

    audit = {'rows_raw': n_raw, 'duplicated_timestamps': n_dup,
             'missing_timestamps': n_missing, 'physically_invalid': n_bad,
             'interpolated_rows': n_nan, 'rows_final': len(df)}
    return df, audit


def add_solar(df):
    loc = pvlib.location.Location(C.LATITUDE, C.LONGITUDE, tz='UTC', altitude=C.ALTITUDE_M)
    sp = loc.get_solarposition(df.index)
    cs = loc.get_clearsky(df.index, model='ineichen', solar_position=sp)
    df['elevation'] = sp['apparent_elevation'].to_numpy()
    df['ghi_cs'] = cs['ghi'].to_numpy()

    kcs = np.where(df['elevation'] > C.KCS_MIN_ELEVATION,
                   df['GHI'] / df['ghi_cs'].clip(lower=1.0), 1.0)
    df['kcs'] = np.clip(kcs, *C.KCS_CLIP)
    df['ghi_cs_next'] = df['ghi_cs'].shift(-1)

    # Calendario en hora local fija (UTC-6), sin horario de verano
    local = df.index.tz_convert(None) + pd.Timedelta(hours=C.UTC_OFFSET_H)
    df['local_time'] = local
    df['hour'] = local.hour
    df['dayofyear'] = local.dayofyear
    df['month'] = local.month
    df['year'] = local.year
    return df


def build_windows(df):
    """X[i] = features de t-24..t-1, y[i] = GHI(t); i indexa el instante objetivo t."""
    feats = df[C.FEATURES].to_numpy(np.float64)
    W = C.WINDOW
    targets = np.arange(W, len(df) - C.HORIZON + 1) + (C.HORIZON - 1)
    # ghi_cs_next del último paso de la ventana existe para todos los objetivos
    idx = targets[:, None] - C.HORIZON + 1 - W + np.arange(W)[None, :]
    return feats, idx, targets


def main():
    os.makedirs(C.DATA_DIR, exist_ok=True)
    df, audit = load_raw()
    df = add_solar(df)
    feats, win_idx, targets = build_windows(df)

    t_local = df['local_time'].to_numpy()[targets]
    t_year = df['year'].to_numpy()[targets]

    split_pos = {}
    for name, years in C.SPLIT_YEARS.items():
        split_pos[name] = np.where(np.isin(t_year, years))[0]

    # Scalers ajustados SOLO con el train principal (filas de entrada usadas por train)
    train_rows = np.unique(win_idx[split_pos['train']].ravel())
    scaler_X = MinMaxScaler().fit(feats[train_rows])
    ghi_train = df['GHI'].to_numpy()[targets[split_pos['train']]].reshape(-1, 1)
    scaler_y = MinMaxScaler().fit(ghi_train)

    feats_s = scaler_X.transform(feats).astype(np.float32)
    X = feats_s[win_idx]                                   # (N, W, F)
    ghi = df['GHI'].to_numpy()
    y = scaler_y.transform(ghi[targets].reshape(-1, 1)).astype(np.float32)

    # Rolling-origin
    folds = []
    t_ts = pd.to_datetime(t_local)
    for k, (start, end) in enumerate(C.ROLLING_FOLDS):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        es_start = start - pd.DateOffset(months=C.ROLLING_ES_MONTHS)
        folds.append({
            'train': np.where(t_ts < es_start)[0],
            'es':    np.where((t_ts >= es_start) & (t_ts < start))[0],
            'test':  np.where((t_ts >= start) & (t_ts < end))[0],
        })

    ghi_t = ghi[targets]
    ghi_cs_t = df['ghi_cs'].to_numpy()[targets]
    elev_t = df['elevation'].to_numpy()[targets]
    day_ghi = ghi_t > C.DAY_THRESHOLD_GHI
    day_elev = elev_t > C.DAY_MIN_ELEVATION

    # Persistencias (usan solo información disponible en t-1)
    prev = targets - 1
    pers = ghi[prev]
    smart = df['kcs'].to_numpy()[prev] * ghi_cs_t
    smart = np.where(elev_t > 0, smart, 0.0)

    np.savez_compressed(
        os.path.join(C.DATA_DIR, 'dataset.npz'),
        X=X, y=y, ghi=ghi_t.astype(np.float32), ghi_cs=ghi_cs_t.astype(np.float32),
        elevation=elev_t.astype(np.float32), day_ghi=day_ghi, day_elev=day_elev,
        time_local=t_local.astype('datetime64[s]'),
        persistence=pers.astype(np.float32), smart_persistence=smart.astype(np.float32),
        **{f'split_{k}': v for k, v in split_pos.items()},
        **{f'fold{i}_{k}': v for i, f in enumerate(folds) for k, v in f.items()},
    )
    joblib.dump(scaler_X, os.path.join(C.DATA_DIR, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(C.DATA_DIR, 'scaler_y.pkl'))

    def span(pos):
        return [str(t_ts[pos[0]]), str(t_ts[pos[-1]]), int(len(pos))]

    summary = {
        'audit': audit,
        'n_sequences': int(len(targets)),
        'X_shape': list(X.shape),
        'features': C.FEATURES,
        'splits': {k: span(v) for k, v in split_pos.items()},
        'rolling_folds': [{k: span(v) for k, v in f.items()} for f in folds],
        'daytime_fraction_ghi50': {k: float(day_ghi[v].mean()) for k, v in split_pos.items()},
        'daytime_fraction_elev5': {k: float(day_elev[v].mean()) for k, v in split_pos.items()},
        'ghi_stats': {
            'all_mean': float(df['GHI'].mean()), 'all_std': float(df['GHI'].std()),
            'max': float(df['GHI'].max()),
            'day_mean': float(df.loc[df['GHI'] > C.DAY_THRESHOLD_GHI, 'GHI'].mean()),
            'day_std': float(df.loc[df['GHI'] > C.DAY_THRESHOLD_GHI, 'GHI'].std()),
            'day_records': int((df['GHI'] > C.DAY_THRESHOLD_GHI).sum()),
        },
    }
    with open(os.path.join(C.DATA_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
