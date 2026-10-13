"""
Ejecuta UNA corrida del benchmark (modelo, optimizador, semilla) en su propio
proceso y termina. Así la memoria que TensorFlow no libera entre modelos se
devuelve al sistema al salir, y una caída solo afecta a esta corrida.

    python run_one.py --model LSTM --opt TPE --seed 3
    python run_one.py --model MLP --opt PSO --seed 0 --pop 20 --tag popsens
    python run_one.py --model GRU --opt DEFAULT --seed 7
    python run_one.py --model MLP --opt GA --seed 0 --dry      # prueba sin TF

Salida (escritura atómica; el .json solo aparece cuando todo terminó):
    outputs/results/<tag>/<run>.json   métricas, historial completo, config
    outputs/results/<tag>/<run>.npz    predicciones de test (W/m²)
"""
import argparse
import json
import os
import platform
import random
import socket
import sys
import time

import numpy as np

import config as C
import hpo

CFG_SNAPSHOT = {k: getattr(C, k) for k in dir(C)
                if k.isupper() and not k.endswith('_DIR') and k != 'RAW_CSV'}


def run_name(model, opt, seed, pop):
    pop_tag = f'_p{pop}' if pop != C.POP_SIZE else ''
    return f'{model}_{opt}{pop_tag}_s{seed:02d}'


# ── Datos ──────────────────────────────────────────────────────────────────────
class Data:
    def __init__(self):
        d = np.load(os.path.join(C.DATA_DIR, 'dataset.npz'))
        self.X, self.y = d['X'], d['y']
        self.ghi = d['ghi']
        self.day_ghi, self.day_elev = d['day_ghi'], d['day_elev']
        self.pers, self.smart = d['persistence'], d['smart_persistence']
        self.split = {k: d[f'split_{k}'] for k in ('train', 'val', 'test')}
        self.folds = [{k: d[f'fold{i}_{k}'] for k in ('train', 'es', 'test')}
                      for i in range(len(C.ROLLING_FOLDS))]
        import joblib
        sy = joblib.load(os.path.join(C.DATA_DIR, 'scaler_y.pkl'))
        self.y_min = float(sy.data_min_[0])
        self.y_range = float(sy.data_range_[0])

    def to_phys(self, y_scaled):
        return np.asarray(y_scaled, dtype=np.float64).ravel() * self.y_range + self.y_min


def metrics(y, p, ref=None):
    e = p - y
    out = {'mae': float(np.mean(np.abs(e))), 'rmse': float(np.sqrt(np.mean(e ** 2))),
           'mbe': float(np.mean(e)),
           'r2': float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}
    for name, r in (ref or {}).items():
        out[f'skill_rmse_{name}'] = float(1 - out['rmse'] / np.sqrt(np.mean((r - y) ** 2)))
        out[f'skill_mae_{name}'] = float(1 - out['mae'] / np.mean(np.abs(r - y)))
    return out


def eval_block(data, pos, pred):
    """Métricas en todas las horas y en las dos definiciones de día."""
    y = data.ghi[pos].astype(np.float64)
    ref = {'pers': data.pers[pos].astype(np.float64),
           'smart': data.smart[pos].astype(np.float64)}
    out = {'all': metrics(y, pred, ref)}
    for name, mask in (('day_ghi50', data.day_ghi[pos]), ('day_elev5', data.day_elev[pos])):
        out[name] = metrics(y[mask], pred[mask], {k: v[mask] for k, v in ref.items()})
    return out


# ── Modelos (misma topología que v1) ───────────────────────────────────────────
def setup_tf():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    n = int(os.environ.get('BENCH_THREADS', 0))
    if n:
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    return tf


def build_model(tf, model_type, p, input_shape):
    L = tf.keras.layers
    if model_type == 'MLP':
        layers = [L.Input(input_shape), L.Flatten(),
                  L.Dense(p['units_1'], activation='relu'), L.Dropout(p['dropout_rate']),
                  L.Dense(p['units_2'], activation='relu'), L.Dropout(p['dropout_rate']),
                  L.Dense(1)]
    else:
        Cell = L.LSTM if model_type == 'LSTM' else L.GRU
        layers = [L.Input(input_shape),
                  Cell(p['units_1'], return_sequences=True), L.Dropout(p['dropout_rate']),
                  Cell(p['units_2']), L.Dropout(p['dropout_rate']),
                  L.Dense(32, activation='relu'), L.Dense(1)]
    m = tf.keras.Sequential(layers)
    m.compile(optimizer=tf.keras.optimizers.Adam(p['learning_rate']), loss='mse',
              metrics=['mae'])
    return m


def set_seed(tf, seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_trainer(tf, data, model_type, seed):
    class EarlyStopRestore(tf.keras.callbacks.Callback):
        """Early stopping sobre val_loss que SIEMPRE restaura los mejores pesos
        y, si se pasa `report`, informa el val MAE (W/m²) por época para poda."""

        def __init__(self, patience, report=None):
            super().__init__()
            self.patience, self.report = patience, report
            self.best, self.best_w, self.wait = np.inf, None, 0
            self.best_mae, self.pruned, self.epochs = np.inf, False, 0

        def on_epoch_end(self, epoch, logs=None):
            self.epochs = epoch + 1
            if logs['val_loss'] < self.best:
                self.best, self.wait = logs['val_loss'], 0
                self.best_mae = logs['val_mae'] * data.y_range
                self.best_w = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True
            if self.report is not None and self.report(epoch + 1, self.best_mae):
                self.pruned = True
                self.model.stop_training = True

        def on_train_end(self, logs=None):
            if self.best_w is not None:
                self.model.set_weights(self.best_w)

    shape = data.X.shape[1:]

    def train(p, tr, es, epochs, patience, report=None):
        tf.keras.backend.clear_session()
        set_seed(tf, seed)
        m = build_model(tf, model_type, p, shape)
        cb = EarlyStopRestore(patience, report)
        m.fit(data.X[tr], data.y[tr], validation_data=(data.X[es], data.y[es]),
              epochs=epochs, batch_size=p['batch_size'], callbacks=[cb], verbose=0,
              shuffle=True)
        if not np.isfinite(cb.best_mae):
            cb.best_mae = 9999.0
        return m, cb

    return train


def predict(m, data, pos):
    return data.to_phys(m.predict(data.X[pos], batch_size=2048, verbose=0))


# ── Corrida ────────────────────────────────────────────────────────────────────
def incumbents(history, opt):
    """Mejor configuración (no podada) disponible a cada presupuesto bajo."""
    out = {}
    for b in C.LOW_BUDGETS + [C.BUDGET]:
        if opt == 'BOHB':
            ok = [h for h in history if h['cum_epochs'] <= b * C.OPT_EPOCHS and not h['pruned']]
        else:
            ok = [h for h in history[:b] if not h['pruned']]
        if ok:
            out[f'B{b}'] = min(ok, key=lambda h: h['val_mae'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=C.MODELS)
    ap.add_argument('--opt', required=True, choices=list(hpo.OPTIMIZERS) + ['DEFAULT'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--pop', type=int, default=C.POP_SIZE)
    ap.add_argument('--tag', default='core')
    ap.add_argument('--budget', type=int, default=C.BUDGET)
    ap.add_argument('--no-rolling', action='store_true')
    ap.add_argument('--dry', action='store_true', help='objetivo sintético, sin TF')
    args = ap.parse_args()

    name = run_name(args.model, args.opt, args.seed, args.pop)
    out_dir = os.path.join(C.RES_DIR, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, name + '.json')
    if os.path.exists(json_path):
        print(f'{name}: ya existe, nada que hacer')
        return

    t_start = time.time()
    data = Data()
    history = []

    if args.dry:
        tf = None
        rng = np.random.default_rng(args.seed)

        def fn(u, report=None):
            p = hpo.decode(u)
            mae = 30 + 40 * (u[2] - 0.05) ** 2 + 30 * (u[3] - 0.6) ** 2 + rng.normal(0, .5)
            epochs, pruned = C.OPT_EPOCHS, False
            if report is not None:
                for e in range(1, C.OPT_EPOCHS + 1):
                    if report(e, mae + 5 / e):
                        epochs, pruned = e, True
                        break
            history.append({'u': [float(x) for x in u], 'params': p, 'val_mae': float(mae),
                            'epochs': epochs, 'pruned': pruned, 'seconds': 0.0})
            return mae, epochs, pruned
    else:
        tf = setup_tf()
        train = make_trainer(tf, data, args.model, args.seed)
        tr, va = data.split['train'], data.split['val']

        def fn(u, report=None):
            p = hpo.decode(u)
            t0 = time.time()
            try:
                _, cb = train(p, tr, va, C.OPT_EPOCHS, C.OPT_PATIENCE, report)
                mae, epochs, pruned = float(cb.best_mae), cb.epochs, cb.pruned
            except Exception as e:  # configuración que falla (p. ej. NaN)
                print(f'  eval fallida {p}: {e}', file=sys.stderr)
                mae, epochs, pruned = 9999.0, 0, False
            history.append({'u': [float(x) for x in u], 'params': p, 'val_mae': mae,
                            'epochs': epochs, 'pruned': pruned,
                            'seconds': round(time.time() - t0, 2)})
            print(f'  eval {len(history):3d}  val_mae={mae:7.2f}  ep={epochs:2d}'
                  f'{"  podada" if pruned else ""}  {time.time() - t0:5.1f}s', flush=True)
            return mae, epochs, pruned

    # ── HPO ──
    t0 = time.time()
    if args.opt == 'DEFAULT':
        incs = {f'B{args.budget}': {'params': dict(C.DEFAULT_PARAMS), 'val_mae': None}}
    else:
        epoch_budget = args.budget * C.OPT_EPOCHS if args.opt == 'BOHB' else None
        n_max = 10 * args.budget if args.opt == 'BOHB' else args.budget
        ev = hpo.Evaluator(fn, n_max, epoch_budget)
        hpo.OPTIMIZERS[args.opt](ev, args.seed, pop=args.pop)
        cum, best = 0, np.inf
        for i, h in enumerate(history):
            cum += h['epochs']
            h['eval'], h['cum_epochs'] = i + 1, cum
            if not h['pruned']:
                best = min(best, h['val_mae'])
            h['best_so_far'] = best
        incs = incumbents(history, args.opt)
    hpo_time = time.time() - t0

    # ── Entrenamientos finales ──
    t0 = time.time()
    finals, preds, done = {}, {}, {}
    # B100 primero: el presupuesto completo siempre tiene su propia evaluación
    # y los presupuestos bajos con el mismo incumbente apuntan a él (same_as).
    for key, inc in sorted(incs.items(), key=lambda kv: -int(kv[0][1:])):
        p = inc['params']
        pkey = json.dumps(p, sort_keys=True)
        entry = {'params': p, 'val_mae_hpo': inc['val_mae'], 'eval': inc.get('eval')}
        if pkey in done:                     # mismo incumbente que otro presupuesto
            entry.update({'same_as': done[pkey]})
            finals[key] = entry
            continue
        if args.dry:
            pred = data.ghi[data.split['test']] + np.random.default_rng(0).normal(
                0, 30, len(data.split['test']))
            epochs = 0
        else:
            m, cb = train(p, data.split['train'], data.split['val'],
                          C.FINAL_EPOCHS, C.FINAL_PATIENCE)
            pred, epochs = predict(m, data, data.split['test']), cb.epochs
            entry['val_mae_final'] = float(cb.best_mae)
        entry.update({'final_epochs': epochs, 'test': eval_block(data, data.split['test'], pred)})
        preds[f'test_{key}'] = pred.astype(np.float32)
        finals[key] = entry
        done[pkey] = key
        print(f'  final {key}: test_mae={entry["test"]["all"]["mae"]:.2f} ep={epochs}', flush=True)

    # Rolling-origin con el incumbente del presupuesto completo
    rolling = None
    if not args.no_rolling:
        p = incs[f'B{args.budget}']['params']
        rolling = {'folds': []}
        all_pos, all_pred = [], []
        for k, f in enumerate(data.folds):
            if args.dry:
                pred, epochs = data.ghi[f['test']].astype(np.float64), 0
            else:
                m, cb = train(p, f['train'], f['es'], C.FINAL_EPOCHS, C.FINAL_PATIENCE)
                pred, epochs = predict(m, data, f['test']), cb.epochs
            preds[f'roll{k}'] = pred.astype(np.float32)
            rolling['folds'].append({'fold': k, 'final_epochs': epochs,
                                     'test': eval_block(data, f['test'], pred)})
            all_pos.append(f['test'])
            all_pred.append(pred)
        rolling['pooled'] = eval_block(data, np.concatenate(all_pos), np.concatenate(all_pred))
    final_time = time.time() - t0

    best = incs[f'B{args.budget}']
    result = {
        'run': name, 'model': args.model, 'optimizer': args.opt, 'seed': args.seed,
        'pop': args.pop, 'tag': args.tag, 'budget': args.budget, 'dry': args.dry,
        'best_params': best['params'], 'best_val_mae': best['val_mae'],
        'best_eval': best.get('eval'),
        'n_evals': len(history), 'n_pruned': int(sum(h['pruned'] for h in history)),
        'total_hpo_epochs': int(sum(h['epochs'] for h in history)),
        'hpo_time_s': round(hpo_time, 1), 'final_time_s': round(final_time, 1),
        'total_time_s': round(time.time() - t_start, 1),
        'finals': finals, 'rolling': rolling, 'history': history,
        'env': {
            'host': socket.gethostname(), 'python': platform.python_version(),
            'tensorflow': getattr(tf, '__version__', None),
            'numpy': np.__version__, 'optuna': hpo.optuna.__version__,
            'gpu': [d.name for d in tf.config.list_physical_devices('GPU')] if tf else [],
            'threads': os.environ.get('BENCH_THREADS'),
            'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'config': CFG_SNAPSHOT,
    }

    npz_tmp = os.path.join(out_dir, name + '.tmp.npz')
    np.savez_compressed(npz_tmp, **preds)
    os.replace(npz_tmp, os.path.join(out_dir, name + '.npz'))
    js_tmp = json_path + '.tmp'
    with open(js_tmp, 'w') as f:
        json.dump(result, f, default=float)
    os.replace(js_tmp, json_path)

    t = finals[f'B{args.budget}']['test']
    print(f'{name}: val={best["val_mae"]} test_mae={t["all"]["mae"]:.2f} '
          f'day_mae={t["day_ghi50"]["mae"]:.2f} evals={len(history)} '
          f'hpo={hpo_time/60:.1f}min final={final_time/60:.1f}min')


if __name__ == '__main__':
    main()
