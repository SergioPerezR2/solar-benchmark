"""
Orquestador del benchmark v2: reparte las corridas en N procesos paralelos,
cada corrida en su propio proceso (`run_one.py`).

- Reanudable: salta las corridas cuyo JSON ya existe.
- Reintenta cada corrida fallida hasta MAX_RETRIES veces.
- Avisa por ntfy al iniciar, cada 5 % de avance, en fallas definitivas y al terminar.
- Al terminar todas las corridas escribe outputs/DONE y sale con código 0
  (con `--restart on-failure` Docker no lo vuelve a lanzar).

    python run_queue.py                 # todo el diseño
    python run_queue.py --max-jobs 6    # piloto: solo las primeras 6 corridas
    python run_queue.py --list          # muestra la cola sin ejecutar
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

import config as C
from run_one import run_name

DONE_FLAG = os.path.join(C.OUT_DIR, 'DONE')
STATE = os.path.join(C.OUT_DIR, 'queue_state.json')


def build_jobs():
    """Cola priorizada. Dentro de cada fase se recorre por semilla para que los
    resultados parciales queden balanceados entre optimizadores, y los modelos
    largos (LSTM/GRU) se lanzan antes que MLP para equilibrar la carga."""
    jobs = []
    order = ['LSTM', 'GRU', 'MLP']

    def add(tag, opts, models, pops=(C.POP_SIZE,)):
        for seed in C.SEEDS:
            for m in order:
                if m not in models:
                    continue
                for o in opts:
                    for p in pops:
                        jobs.append({'tag': tag, 'model': m, 'opt': o, 'seed': seed, 'pop': p})

    add('core', ['DEFAULT'], C.MODELS)
    add('core', C.CORE_OPTIMIZERS, C.MODELS)
    add('core', C.EXTRA_OPTIMIZERS, C.MODELS)
    add('popsens', C.POP_SENS_OPTIMIZERS, C.POP_SENS_MODELS, C.POP_SENS_SIZES)
    for j in jobs:
        j['name'] = run_name(j['model'], j['opt'], j['seed'], j['pop'])
        j['json'] = os.path.join(C.RES_DIR, j['tag'], j['name'] + '.json')
    return jobs


def notify(msg):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)
    if not C.NTFY_TOPIC:
        return
    try:
        import requests
        requests.post(f'https://ntfy.sh/{C.NTFY_TOPIC}', data=msg.encode(), timeout=10)
    except Exception:
        pass


def fmt_h(s):
    h, r = divmod(int(s), 3600)
    return f'{h}h{r // 60:02d}m'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=C.N_WORKERS)
    ap.add_argument('--max-jobs', type=int, default=None)
    ap.add_argument('--filter', default=None, help='regex sobre "<tag>:<run>" (piloto)')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    os.makedirs(C.LOG_DIR, exist_ok=True)
    jobs = build_jobs()
    if args.filter:
        import re
        jobs = [j for j in jobs if re.search(args.filter, f"{j['tag']}:{j['name']}")]
    if args.max_jobs:
        jobs = jobs[:args.max_jobs]
    partial = bool(args.filter or args.max_jobs)
    pending = deque(j for j in jobs if not os.path.exists(j['json']))
    total, n_done0 = len(jobs), len(jobs) - len(pending)

    if args.list:
        for j in pending:
            print(j['tag'], j['name'])
        print(f'{len(pending)} pendientes de {total}')
        return

    if not pending:
        if not partial and not os.path.exists(DONE_FLAG):
            with open(DONE_FLAG, 'w') as f:
                json.dump({'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
                           'done': total, 'total': total, 'failed': []}, f, indent=1)
        print('Nada pendiente.')
        return
    if not partial and os.path.exists(DONE_FLAG):
        print('outputs/DONE existe: la cola ya terminó. Bórralo para relanzar.')
        return

    threads = max(1, (os.cpu_count() or 8) // args.workers)
    env = dict(os.environ, BENCH_THREADS=str(threads), TF_CPP_MIN_LOG_LEVEL='3')
    notify(f'🟢 Benchmark v2 iniciado: {n_done0}/{total} hechas, {len(pending)} pendientes, '
           f'{args.workers} workers x {threads} hilos.')

    running, tries, failed, durations = {}, {}, [], []
    t0, last_pct = time.time(), -1

    while pending or running:
        while pending and len(running) < args.workers:
            j = pending.popleft()
            log = open(os.path.join(C.LOG_DIR, j['name'] + ('_' + j['tag'] if j['tag'] != 'core' else '') + '.log'), 'a')
            cmd = [sys.executable, '-u', 'run_one.py', '--model', j['model'], '--opt', j['opt'],
                   '--seed', str(j['seed']), '--pop', str(j['pop']), '--tag', j['tag']]
            proc = subprocess.Popen(cmd, cwd=C.V2_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[proc.pid] = (proc, j, time.time(), log)

        time.sleep(5)
        for pid, (proc, j, ts, log) in list(running.items()):
            rc = proc.poll()
            if rc is None and time.time() - ts > C.JOB_TIMEOUT_S:
                proc.kill()
                rc = proc.wait()
            if rc is None:
                continue
            log.close()
            del running[pid]
            if rc == 0 and os.path.exists(j['json']):
                durations.append(time.time() - ts)
            else:
                tries[j['name']] = tries.get(j['name'], 0) + 1
                if tries[j['name']] <= C.MAX_RETRIES:
                    notify(f'⚠️ {j["name"]} falló (código {rc}), reintento {tries[j["name"]]}.')
                    pending.append(j)
                else:
                    failed.append(j['name'])
                    notify(f'🔴 {j["name"]} falló {C.MAX_RETRIES + 1} veces; se omite.')

        n_done = sum(os.path.exists(j['json']) for j in jobs)
        pct = int(100 * n_done / total)
        if pct // 5 > last_pct // 5 and durations:
            last_pct = pct
            elapsed = time.time() - t0
            rate = (n_done - n_done0) / elapsed if elapsed > 0 else 0
            eta = (total - n_done) / rate if rate > 0 else 0
            notify(f'📊 v2: {n_done}/{total} ({pct}%). Transcurrido {fmt_h(elapsed)}, '
                   f'faltan ~{fmt_h(eta)}.')
        with open(STATE, 'w') as f:
            json.dump({'done': n_done, 'total': total, 'running': [r[1]['name'] for r in running.values()],
                       'pending': len(pending), 'failed': failed,
                       'updated': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=1)

    # DONE se escribe aunque haya fallas definitivas: así ni Docker ni el
    # watchdog relanzan la cola en bucle; las fallidas se revisan a mano.
    n_done = sum(os.path.exists(j['json']) for j in jobs)
    if not partial:
        with open(DONE_FLAG, 'w') as f:
            json.dump({'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'done': n_done,
                       'total': total, 'failed': failed}, f, indent=1)
    if failed:
        notify(f'🟠 v2 terminó con fallas: {n_done}/{total}. Fallidas: {", ".join(failed)}')
    else:
        notify(f'✅ v2 COMPLETADO: {n_done}/{total} corridas en {fmt_h(time.time() - t0)}.')


if __name__ == '__main__':
    main()
