# Imagen del benchmark v2: la misma de Jupyter (TF + GPU ya probados) más
# pvlib (cielo despejado Ineichen) y cmaes (muestreador CMA-ES de Optuna).
FROM jupyter-gpu:bench-base
USER root
RUN pip install --no-cache-dir "pvlib>=0.13" "cmaes>=0.11"
USER jovyan
WORKDIR /work/v2
CMD ["python", "-u", "run_queue.py"]
