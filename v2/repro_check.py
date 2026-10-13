"""Comprueba la reproducibilidad de corridas repetidas con la misma semilla.

Las corridas de outputs/results/repro/ repiten corridas de outputs/results/core/
con el mismo código y la misma semilla. Para cada par se reporta desde qué
evaluación difieren las configuraciones propuestas, cuánto difiere el MAE de
validación de una misma configuración (no determinismo de los kernels de GPU),
si coincide el incumbente y la diferencia en el MAE de test.

Uso:  python repro_check.py
    (genera las corridas con: python run_one.py --model MLP --opt PSO --seed 0 --tag repro)
"""
import glob
import json
import os

import numpy as np

import config as C

rows = []
for f in sorted(glob.glob(os.path.join(C.RES_DIR, 'repro', '*.json'))):
    a = json.load(open(f))
    b = json.load(open(os.path.join(C.RES_DIR, 'core', os.path.basename(f))))
    ha, hb = a['history'], b['history']
    k = min(len(ha), len(hb))
    same = [np.allclose(ha[i]['u'], hb[i]['u']) for i in range(k)]
    first_cfg = next((i + 1 for i, v in enumerate(same) if not v), None)
    d = np.array([abs(ha[i]['val_mae'] - hb[i]['val_mae']) for i in range(k) if same[i]])
    ta = a['finals']['B100']['test']['all']['mae']
    tb = b['finals']['B100']['test']['all']['mae']
    rows.append((os.path.basename(f)[:-5], first_cfg, np.median(d), 100 * np.mean(d < 1e-6),
                 a['best_params'] == b['best_params'], ta - tb))
    print(f'{rows[-1][0]:16s} configuraciones distintas desde la evaluación {first_cfg}; '
          f'misma configuración: |dMAE val| mediana {np.median(d):.2f}, idénticas {100 * np.mean(d < 1e-6):.0f}%; '
          f'mismo incumbente: {rows[-1][4]}; dMAE test {ta - tb:+.2f} W/m²')
dt = np.abs([r[5] for r in rows])
print(f'|dMAE test|: {dt.min():.2f}-{dt.max():.2f} W/m² (mediana {np.median(dt):.2f})')
