"""
Envía por ntfy un resumen del avance con una estimación realista del tiempo
restante. A diferencia del aviso de la cola, estima por tipo de corrida
(modelo y método) con las duraciones reales ya medidas, así que no se deja
engañar por las corridas DEFAULT, que son muy rápidas.

Pensado para cron en el host (solo librería estándar):
    0 */3 * * * python3 ~/solar-benchmark/v2/status_notify.py
"""
import glob
import json
import os
import urllib.request

V2 = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(V2, 'outputs')
TOPIC = os.environ.get('NTFY_TOPIC', '')
WORKERS = 6

if os.path.exists(os.path.join(OUT, 'DONE')):
    raise SystemExit
state = json.load(open(os.path.join(OUT, 'queue_state.json')))

dur = {}
for p in glob.glob(os.path.join(OUT, 'results', '*', '*.json')):
    r = json.load(open(p))
    kind = 'DEFAULT' if r['optimizer'] == 'DEFAULT' else 'HPO'
    dur.setdefault((r['model'], kind), []).append(r['total_time_s'])
mean = {k: sum(v) / len(v) for k, v in dur.items()}

# Corridas que faltan por tipo: 7 métodos x 3 modelos x 30 semillas + 180 de población (MLP)
done = {}
for p in glob.glob(os.path.join(OUT, 'results', '*', '*.json')):
    r = json.load(open(p))
    kind = 'DEFAULT' if r['optimizer'] == 'DEFAULT' else 'HPO'
    done[(r['model'], kind)] = done.get((r['model'], kind), 0) + 1
total = {(m, 'DEFAULT'): 30 for m in ('MLP', 'LSTM', 'GRU')}
total.update({(m, 'HPO'): 210 for m in ('MLP', 'LSTM', 'GRU')})
total[('MLP', 'HPO')] += 180
fallback = {'MLP': 2400, 'LSTM': 6200, 'GRU': 4600}
left_s = sum((total[k] - done.get(k, 0)) * mean.get(k, fallback[k[0]]) for k in total)
eta_h = left_s / WORKERS / 3600

msg = (f'⏱ v2: {state["done"]}/{state["total"]} corridas ({100 * state["done"] / state["total"]:.0f}%). '
       f'Faltan ~{eta_h / 24:.1f} días ({eta_h:.0f} h). En curso: {", ".join(state["running"])}. '
       f'Fallidas: {len(state["failed"])}.')
if TOPIC:
    urllib.request.urlopen(urllib.request.Request(f'https://ntfy.sh/{TOPIC}', data=msg.encode()), timeout=15)
else:
    print(msg)
print(msg)
