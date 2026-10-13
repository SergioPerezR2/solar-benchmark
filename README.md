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
  surrogate_benchmark.py     Surrogate benchmark fitted to all evaluations (see below)
  robustness.py              Robustness checks: night-clipped and daytime errors, omega² with F tests,
                             validation-test agreement of the retrained models, shared initial designs,
                             replay of the ABC scout phase
  repro_check.py             Compares runs repeated with the same seed (GPU non-determinism)
  figures/                   Scripts for all figures (results_figures.py, f6, i1, i2)
                             and for all tables and numbers in the text (results_tables.py)
  outputs/
    data/                    Preprocessed dataset and scalers
    results/core/            720 runs: <MODEL>_<OPTIMIZER>_s<SEED>.json and .npz
    results/popsens/         180 population-size sensitivity runs
    analysis/                Analysis tables (CSV), tables.tex, and numbers.json
    surrogate/               Surrogate models, fidelity, and replay of the optimizers
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
   python robustness.py
   python figures/results_tables.py
   python figures/results_figures.py
   python figures/f6_anytime_convergence.py
   ```

## Surrogate benchmark: test your own optimizer in seconds

### What problem it solves

Comparing hyperparameter optimizers on a real deep learning task is expensive. In this study, one run (100 evaluations, each one training a network) took 27 to 83 minutes of GPU time, and a fair comparison needs about 30 runs per optimizer and architecture. Testing one new optimizer the way the paper does (30 runs on each of the three architectures) would take about 80 hours of computing for the searches alone, and about 100 hours including the final trainings, with several runs sharing one GPU.

The surrogate benchmark replaces the expensive part (training a network for each configuration) with a fast model that predicts the result that training would give. With it, a new optimizer can be compared with the seven optimizers of the paper on the same task, under the same protocol, in about 30 seconds on a laptop.

### How it works, in plain terms

1. **Data.** During the experiment, 72,457 networks were trained. The surrogate uses the 59,151 trainings that ran to completion: 54,000 from the six optimizers with a fixed number of evaluations (6 optimizers × 3 architectures × 30 seeds × 100 evaluations) and 5,151 complete trials of the BOHB-style method. The other 13,306 BOHB-style trials were stopped early by Hyperband and are excluded, because they do not give the error of a complete training. For each training, the configuration and its validation error are stored. About 9,000 of them come from Random Search, which samples the whole search space uniformly, so every region of the space is represented; the rest come from the guided optimizers and are more dense in the good regions.
2. **Prediction model.** For each architecture (MLP, LSTM, GRU), a gradient-boosting regression model learns to map a configuration (units of the two layers, dropout, learning rate, batch size; encoded in [0, 1]^5 exactly as in the experiment) to its validation MAE.
3. **Training noise.** Training the same configuration with another seed gives a slightly different error. The surrogate adds noise of the size observed in the experiment: a shift common to the whole run (the effect of the seed) and a small random variation per evaluation. The size of the noise was tuned with Random Search only, so that Random Search on the surrogate reaches the same error as in the real experiment.
4. **Fair comparison.** When a new optimizer is run, it receives exactly the same noise as the paper's optimizers for each seed, so the comparison is paired seed by seed, as in the paper.

### How we know it is reliable

The six optimizers of the paper that do not use learning curves (PSO, GA, ABC, CMA-ES, TPE, Random Search) were run again on the surrogate, with the same code and seeds, and compared with the real experiment (files in `v2/outputs/surrogate/`):

| Check | Result |
|---|---|
| Rank correlation between predicted and real validation error, for seeds not used to fit the model | 0.62 (GRU), 0.69 (LSTM), 0.85 (MLP) |
| Mean difference between the real and the surrogate result of the five optimizers not used for calibration (after 10, 25, 50, and 100 evaluations) | 0.25 W/m² (largest 1.03 W/m²) |
| Rank correlation of all optimizers, architectures, and budgets, real vs. surrogate | 0.96 |
| Worst optimizer after 100 evaluations | Random Search in all three architectures, in the experiment and on the surrogate |
| Worst optimizer after 10 evaluations on the MLP | CMA-ES, in the experiment and on the surrogate |
| Time per run of 100 evaluations | 0.2 s (surrogate) vs. 27–83 min (real training) |

As an additional check, if Random Search is passed to the comparison function as if it were a new method, the difference with the paper's Random Search is exactly 0.00 W/m², which confirms that the comparison is paired correctly.

### Quick start: compare your optimizer with the paper's optimizers

Write your optimizer as a Python function that receives `objective`, `budget`, and `seed`. `objective(u)` takes a vector `u` with 5 values in [0, 1] and returns the validation MAE in W/m² (lower is better). When the budget of 100 evaluations is used up, `objective` stops the run automatically.

```python
# run from the v2/ folder
from scipy.optimize import differential_evolution
import surrogate_benchmark as S

def my_optimizer(objective, budget, seed):
    differential_evolution(objective, bounds=[(0, 1)] * 5, popsize=2, maxiter=10_000,
                           init='random', polish=False, tol=0, seed=seed)

summary, comparison = S.compare_new_optimizer(my_optimizer, name='DE')
```

`examples/try_new_optimizer.py` contains this example. It evaluates Differential Evolution, an optimizer that is **not** part of the paper, on the three architectures with 30 seeds in about 30 seconds:

```bash
cd v2 && python examples/try_new_optimizer.py
```

Part of its output (difference between DE and each optimizer of the paper after 100 evaluations, in W/m²; negative means DE found a lower validation error):

| Architecture | vs. | Mean difference | 90% interval | Better | Worse | Equivalent within 1 W/m² |
|---|---|---|---|---|---|---|
| MLP | RS | −1.05 | [−1.33, −0.77] | yes | no | no |
| MLP | PSO | +0.31 | [−0.05, 0.63] | no | no | yes |
| LSTM | RS | −0.56 | [−0.75, −0.37] | yes | no | yes |
| LSTM | TPE | +0.24 | [0.12, 0.36] | no | yes | yes |
| GRU | ABC | −0.22 | [−0.34, −0.07] | yes | no | yes |

### How to read the results

- `summary`: mean best validation MAE of each optimizer after 10, 25, 50, and 100 evaluations, per architecture.
- `comparison`: for your optimizer against each optimizer of the paper, the mean difference paired by seed, its 90% bootstrap interval, and three flags:
  - `better` / `worse`: the interval lies entirely below / above zero;
  - `equivalent_d1`: the interval lies within ±1 W/m², the practical margin used in the paper. A difference can be "better" and still "equivalent", which means it exists but is too small to matter in practice.

In the example, DE is better than Random Search and practically equivalent to the other optimizers, the same pattern found in the paper for all guided methods.

### Limitations and biases: what the surrogate can and cannot tell you

The surrogate reproduces **this** experiment. It is a realistic and cheap test problem, not a general truth about optimizers.

- **One task.** It represents one-hour-ahead GHI forecasting at one site (Querétaro, México, PVGIS-ERA5), with these three architectures and this five-dimensional search space. It says nothing about other datasets, horizons, architectures, or hyperparameters (for example, number of layers). A method that works well here is not guaranteed to work well elsewhere.
- **Validation error, not test error.** It predicts the error on the validation year (2021), which is what optimizers minimize. The paper shows that better validation error does not always translate into better test error; claims about final accuracy still require real training.
- **It slightly favours model-based methods such as TPE**, which can exploit regularities of the regression model. Its results are most reliable for screening (discarding clearly bad ideas, tuning a new method, finding obvious failures), not for deciding a winner by a few tenths of W/m².
- **Accuracy is lower among the best configurations**, where differences between configurations are smaller than the training noise (also true for real training).
- **No learning curves.** Multi-fidelity methods that stop trainings early (such as BOHB) cannot be evaluated, because the surrogate only predicts the result of a complete evaluation.

Recommended use: develop and screen new optimizers on the surrogate, then confirm the final comparison with real training on a few architectures and seeds.

### Advanced use

The objective can also be used directly:

```python
import numpy as np
from surrogate_benchmark import SurrogateObjective
import hpo

f = SurrogateObjective('LSTM', seed=0)       # noise=False gives the noise-free prediction
u = np.random.default_rng(0).random(5)
print(hpo.decode(u), f(u))                    # decoded hyperparameters and validation MAE (W/m2)
```

`python surrogate_benchmark.py` refits the surrogate from the released results and repeats the validation against the experiment.

## What the comparison designs used in the literature would conclude

`analyze.py` (function `literature_designs`) resamples the 30-seed results to emulate studies with 1 to 30 runs per optimizer, the designs used in the literature. Each study draws, independently for each optimizer, n of its 30 results with replacement (a plug-in bootstrap: the observed distribution of each optimizer acts as the population). Every quantity is also computed under the null hypothesis of equal means, by centering the results of each optimizer on the grand mean; the null value is given in parentheses. Main results (range over the three architectures, `v2/outputs/analysis/literature_designs.csv`):

| Runs per optimizer | Apparent winner is the true best | Two independent studies agree on the winner | A pre-chosen method looks significantly better than a rival (uncorrected tests) | Same, with Holm correction | Uncorrected, with a difference larger than 1 W/m² |
|---|---|---|---|---|---|
| 1 | 19–22% (14%) | 14–15% (15–16%) | – | – | – |
| 15 | 33–55% (14%) | 22–35% (15–16%) | 18–30% (10–12%) | 2–4% (0–1%) | 0–9% (0–1%) |
| 30 | 44–74% (14%) | 27–55% (15–16%) | 27–45% (13–14%) | 4–14% (1–2%) | 0–3% (0%) |

Because the seven optimizers are practically equivalent on this task, any "winner" is mostly due to chance. About half of the uncorrected "significant improvements" are the false positives expected among equivalent methods, and most of the rest are differences below 1 W/m². More runs alone do not fix this; correcting for multiple comparisons and requiring a practical margin do.

## Robustness and reproducibility checks

`robustness.py` repeats the optimizer comparison with three other error definitions (all hours with night clipping, daytime by GHI threshold, daytime by solar elevation). No pair differs after Holm correction with any definition, and 60–63 of the 63 pairs are equivalent within 1 W/m². It also shows that the weak validation–test agreement comes from the 10-epoch HPO objective, not from the change of year: the validation MAE of the retrained final models has a Spearman correlation of 0.98–0.99 with their test MAE.

`repro_check.py` compares six runs repeated with the same seed (`run_one.py ... --tag repro`). GPU kernels are not deterministic, so the same configuration does not give exactly the same validation MAE (the difference is exactly zero in about half of the MLP evaluations and never for LSTM/GRU). The repeated runs selected different incumbents, and their test MAE changed by 0.01–4.04 W/m² (median 0.44 W/m²).

## License

Code released under the [MIT License](LICENSE). Data usage is subject to the terms of the original providers (PVGIS/ERA5, European Commission Joint Research Centre, Copernicus).
