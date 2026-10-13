#!/usr/bin/env bash
# Vigila el contenedor del benchmark v2 y lo vuelve a arrancar si se cae
# (p. ej., tras reiniciar el servidor). TERMINA en cuanto la cola acaba:
#   - existe v2/outputs/DONE, o
#   - el contenedor salió con código 0 (corrida piloto con --max-jobs).
# Uso:  nohup ~/watch_benchmark.sh >> ~/watch_benchmark.log 2>&1 &
CONTAINER="solar-bench"
DONE="$HOME/solar-benchmark/v2/outputs/DONE"

while true; do
  if [ -f "$DONE" ]; then
    echo "$(date) cola terminada (DONE). Watchdog finaliza."
    exit 0
  fi

  status=$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "$CONTAINER" 2>/dev/null || echo "missing -1")
  case "$status" in
    running*) ;;
    "exited 0")
      echo "$(date) $CONTAINER terminó con código 0. Watchdog finaliza."
      exit 0 ;;
    missing*)
      echo "$(date) no existe el contenedor $CONTAINER; lánzalo con v2/launch.sh. Watchdog finaliza."
      exit 1 ;;
    *)
      echo "$(date) $CONTAINER caído ($status), arrancando..."
      docker start "$CONTAINER" >/dev/null ;;
  esac
  sleep 60
done
