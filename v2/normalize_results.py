"""
Normaliza resultados escritos por la primera versión de run_one.py, en la que
un presupuesto bajo podía quedarse con la evaluación y B100 con `same_as`.
Deja siempre la evaluación en el presupuesto mayor. Es idempotente.

    python normalize_results.py
"""
import glob
import json
import os

import numpy as np

import config as C

fixed = 0
for path in glob.glob(os.path.join(C.RES_DIR, '*', '*.json')):
    r = json.load(open(path))
    fin = r['finals']
    keys = sorted(fin, key=lambda k: -int(k[1:]))
    changed = False
    # Para cada grupo de presupuestos con el mismo incumbente, mover la
    # evaluación al presupuesto mayor del grupo.
    groups = {}
    for k in keys:
        groups.setdefault(json.dumps(fin[k]['params'], sort_keys=True), []).append(k)
    npz_path = path[:-5] + '.npz'
    preds = dict(np.load(npz_path)) if os.path.exists(npz_path) else {}
    for ks in groups.values():
        top = ks[0]
        owner = next(k for k in ks if 'test' in fin[k])
        if owner != top:
            for f in ('test', 'final_epochs', 'val_mae_final'):
                if f in fin[owner]:
                    fin[top][f] = fin[owner].pop(f)
            fin[top].pop('same_as', None)
            if f'test_{owner}' in preds:
                preds[f'test_{top}'] = preds.pop(f'test_{owner}')
            changed = True
        for k in ks[1:]:
            if fin[k].get('same_as') != top:
                fin[k]['same_as'] = top
                changed = True
    if changed:
        tmp = npz_path[:-4] + '.tmp.npz'
        np.savez_compressed(tmp, **preds)
        os.replace(tmp, npz_path)
        with open(path + '.tmp', 'w') as f:
            json.dump(r, f, default=float)
        os.replace(path + '.tmp', path)
        fixed += 1
print(f'{fixed} resultados normalizados')
