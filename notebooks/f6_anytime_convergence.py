"""
Figure: Anytime convergence curves — mean ± std best validation MAE vs HPO
evaluation step, per optimizer, faceted by model architecture.

Data source: MLflow metrics extracted via extract_convergence.py
Run: python notebooks/f6_anytime_convergence.py
Output: /tmp/f6_anytime_convergence.pdf
"""
import pathlib
import csv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Extract convergence data from MLflow ─────────────────────────────────────
MLFLOW_DIR = pathlib.Path("/home/sergioperez/solar-benchmark/mlruns/472636885602395698")
OUT_CSV    = pathlib.Path("/tmp/convergence_curves.csv")

rows = []
for run_dir in MLFLOW_DIR.iterdir():
    tags   = run_dir / "tags"
    metric = run_dir / "metrics" / "anytime_best_mae"
    if not (tags.exists() and metric.exists()):
        continue
    model     = (tags / "model").read_text().strip()
    optimizer = (tags / "optimizer").read_text().strip()
    seed      = int((tags / "seed").read_text().strip())
    for line in metric.read_text().splitlines():
        ts, val, step = line.split()
        rows.append({"model": model, "optimizer": optimizer, "seed": seed,
                     "eval_step": int(step), "best_val_mae": float(val)})

df = pd.DataFrame(rows)

# Deduplicate and enforce monotonic decrease within each run
df = df.groupby(["model", "optimizer", "seed", "eval_step"], as_index=False)["best_val_mae"].min()
df = df.sort_values(["model", "optimizer", "seed", "eval_step"])
df["best_val_mae"] = df.groupby(["model", "optimizer", "seed"])["best_val_mae"].cummin()
df = df[df["eval_step"] <= 100]

# ── Aggregate ─────────────────────────────────────────────────────────────────
agg = (
    df.groupby(["model", "optimizer", "eval_step"])["best_val_mae"]
    .agg(mean="mean", std="std")
    .reset_index()
)

# ── Plot ───────────────────────────────────────────────────────────────────────
MODELS    = ["MLP", "LSTM", "GRU"]
OPTS      = ["PSO", "GA", "ABC", "TPE"]
COLORS    = {"PSO": "#1f77b4", "GA": "#ff7f0e", "ABC": "#2ca02c", "TPE": "#d62728"}
LINESTYLE = {"PSO": "-",       "GA": "--",      "ABC": "-.",      "TPE": ":"}

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)

for ax, model in zip(axes, MODELS):
    sub = agg[agg["model"] == model]
    for opt in OPTS:
        s  = sub[sub["optimizer"] == opt]
        x  = s["eval_step"].values
        mu = s["mean"].values
        sd = s["std"].values
        ax.plot(x, mu, label=opt, color=COLORS[opt], ls=LINESTYLE[opt], lw=1.8)
        ax.fill_between(x, mu - sd, mu + sd, color=COLORS[opt], alpha=0.12)
    ax.set_title(model, fontsize=11, fontweight="bold")
    ax.set_xlabel("HPO evaluation", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.set_xlim(1, 100)

axes[0].set_ylabel("Best val. MAE (W/m²)", fontsize=9)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = "/tmp/f6_anytime_convergence.pdf"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
