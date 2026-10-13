#!/usr/bin/env bash
# Construye la imagen y lanza la cola v2 en un contenedor dedicado.
#   ./v2/launch.sh                 # diseño completo
#   ./v2/launch.sh --max-jobs 6    # piloto
# Seguimiento:  docker logs -f solar-bench   |   cat v2/outputs/queue_state.json
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS="${BENCH_WORKERS:-6}"

docker build -q -t solar-bench:v2 v2/ >/dev/null
docker rm -f solar-bench >/dev/null 2>&1 || true
docker run -d --name solar-bench \
  --gpus all \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  --memory 26g --memory-swap 26g --shm-size 2g \
  --restart on-failure:20 \
  -e BENCH_WORKERS="$WORKERS" \
  -v "$PWD":/work -w /work/v2 \
  solar-bench:v2 python -u run_queue.py "$@"
echo "Lanzado solar-bench con $WORKERS workers. Logs: docker logs -f solar-bench"
