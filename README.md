# solar-benchmark

Code, data, and results for the article *"Comparative Benchmark of Metaheuristic and Model-Based Hyperparameter Optimization for Deep Learning Models in Global Solar Irradiance Forecasting"*.

The repository has two parts:

- **`v2/`**: the revised benchmark used in the current version of the article. It contains the code, the preprocessed data, and the complete results of all 900 runs (configuration, evaluation history, metrics, and test-set predictions), together with the scripts that generate every table and figure of the article.
- **Root (`notebooks/`, `run_benchmark.sh`)**: the original version of the benchmark (v1), kept for reference.

## Benchmark (v2)

Seven hyperparameter optimization (HPO) methods are compared on three deep learning architectures for one-hour-ahead forecasting of global horizontal irradiance (GHI):

- **Optimizers:** particle swarm optimization (PSO), genetic algorithm (GA), artificial bee colony (ABC), CMA-ES, tree-structured Parzen estimator (TPE), a BOHB-style method (TPE with Hyperband pruning), and random search (RS).
- **Models:** MLP, LSTM, and GRU.
- **Data:** PVGIS-ERA5 hourly record for Querétaro, México (lat 20.563425, lon −100.36943889), 2019–2022. Training 2019–2020, validation 2021, test 2022, plus a rolling-origin evaluation over the four quarters of 2022.
- **Inputs:** 24-hour window of GHI, clear-sky index, air temperature, wind speed, hour, day of year, month, and the clear-sky GHI of the following hour (Ineichen model, pvlib).
- **Design:** shared search space encoded in [0, 1]^5, budget of 100 evaluations (1,000 training epochs for the BOHB-style method), and 30 paired seeds per optimizer and architecture. Incumbents after 10, 25, 50, and 100 evaluations are evaluated on the test year.
- **Runs:** 630 HPO runs, 90 runs with default hyperparameters, and 180 population-size sensitivity runs (PSO, GA, and ABC on the MLP with populations of 5 and 20).
- **Analysis:** Friedman tests, paired Wilcoxon tests with Holm correction, BCa bootstrap intervals, TOST equivalence tests (δ = 1 W/m², with 0.5 and 2 as sensitivity), Vargha–Delaney Â12, minimum detectable difference, variance decomposition, and an emulation of single-run comparisons.

### Structure

```
data/download_pvgis.py       Downloads the PVGIS-ERA5 record to data/pvgis_raw.csv
v2/
  config.py                  All settings of the experiment (stored in every result file)
  prepare_data.py            Builds outputs/data/dataset.npz (features, splits, folds, references)
  hpo.py                     Shared encoding and the seven optimizers
  run_one.py                 One run (HPO + final models + rolling-origin) in its own process
  run_queue.py               Runs the 900 jobs in parallel with retries
  launch.sh, Dockerfile      Container launcher used for the experiments
  normalize_results.py       Consistency pass over the result files
  analyze.py                 Statistical analysis -> outputs/analysis/*.csv
  symbolic_refit.py          Re-fit and bootstrap of the symbolic reference equation
  figures/                   Scripts for all figures (results_figures.py, f6, i1, i2)
                             and for all tables and numbers in the text (results_tables.py)
  outputs/
    data/                    Preprocessed dataset and scalers
    results/core/            720 runs: <MODEL>_<OPTIMIZER>_s<SEED>.json and .npz
    results/popsens/         180 population-size sensitivity runs
    analysis/                Analysis tables (CSV), tables.tex, and numbers.json
```

Each result file `*.json` contains the configuration, the full evaluation history (candidate, validation MAE, epochs, and time of every evaluation), the incumbents and test metrics at b = 10, 25, 50, and 100, and the rolling-origin results. The matching `*.npz` file contains the test-set predictions.

### Reproducing the results

1. **Environment** (Python 3.12; a GPU is recommended):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r v2/requirements.txt
   ```
2. **Data:**
   ```bash
   python data/download_pvgis.py
   cd v2 && python prepare_data.py
   ```
3. **Experiment** (all 900 runs; completed runs are skipped, so the queue can be resumed):
   ```bash
   BENCH_WORKERS=4 python run_queue.py
   python run_one.py --model LSTM --opt TPE --seed 0     # a single run
   ```
   The experiments of the paper were run with `launch.sh` in a Docker container (TensorFlow 2.21 on an NVIDIA RTX 4070); the versions are listed in `v2/requirements.txt`. Progress notifications through ntfy.sh are optional (`NTFY_TOPIC` environment variable).
4. **Analysis, tables, and figures** (from the released results, without retraining):
   ```bash
   python normalize_results.py
   python analyze.py --boot 10000
   python symbolic_refit.py --boot 300
   python figures/results_tables.py
   python figures/results_figures.py
   python figures/f6_anytime_convergence.py
   ```

## License

Code released under the [MIT License](LICENSE). Data usage is subject to the terms of the original providers (PVGIS/ERA5, European Commission Joint Research Centre, Copernicus).
