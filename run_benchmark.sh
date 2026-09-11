#!/bin/bash
TOPIC="sergio-benchmark-uaq"
LOG="/home/jovyan/work/benchmark.log"
NB="/home/jovyan/work/notebooks/03_benchmark.ipynb"
OUT="/home/jovyan/work/notebooks/03_benchmark_output.ipynb"
RESULTS="/home/jovyan/work/outputs/pvgis/benchmark/results/benchmark_results.csv"
TOTAL=360

notify() {
    python3 - "$TOPIC" "$1" <<'PY' 2>/dev/null
import sys, requests
topic, msg = sys.argv[1], sys.argv[2]
requests.post(f"https://ntfy.sh/{topic}", data=msg.encode())
PY
}

progress_msg() {
python3 - "$RESULTS" "$TOTAL" "$1" <<'PY'
import sys, os, time, pandas as pd

f, total, start = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
done = len(pd.read_csv(f)) if os.path.exists(f) else 0
elapsed = time.time() - start
rate = done / elapsed if elapsed > 0 else 0
left = max(total - done, 0)
eta = left / rate if rate > 0 else 0

def fmt(sec):
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s"

pct = 100 * done / total if total else 0
print(
    f"Progreso: {done}/{total} runs ({pct:.1f}%). "
    f"Tiempo: {fmt(elapsed)}. "
    f"Ritmo: {rate*3600:.2f} runs/h. "
    f"Faltan aprox: {fmt(eta)}."
)
PY
}

START=$(date +%s)

echo "[$(date +"%Y-%m-%d %H:%M")] Iniciando benchmark" | tee -a "$LOG"
notify "🟢 Benchmark iniciado en athenacore. $(progress_msg "$START")"

jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output "$OUT" \
    "$NB" >> "$LOG" 2>&1

EXIT=$?
MSG="$(progress_msg "$START")"

if [ $EXIT -eq 0 ]; then
    notify "✅ COMPLETADO. $MSG Revisa MLflow."
else
    LASTERR=$(tail -40 "$LOG" | grep -E "DeadKernelError|OutOfMemory|Killed|Traceback|Error|Exception" | tail -5)
    notify "🔴 CRASHEO exit $EXIT. $MSG Último error: ${LASTERR:-ver benchmark.log}"
fi

exit $EXIT
