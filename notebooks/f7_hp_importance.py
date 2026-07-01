"""
Figure: Hyperparameter importance via permutation importance on a random
forest surrogate fitted to (hp_* -> best_val_mae), per model architecture.

Data source: outputs/pvgis/benchmark/results/benchmark_results.csv
Run: python notebooks/f7_hp_importance.py
Output: /tmp/f7_hp_importance.pdf
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

# ── Data ───────────────────────────────────────────────────────────────────────
CSV = "/home/sergioperez/solar-benchmark/outputs/pvgis/benchmark/results/benchmark_results.csv"
df  = pd.read_csv(CSV)

HP_COLS   = ["hp_units_1", "hp_units_2", "hp_dropout_rate", "hp_learning_rate", "hp_batch_size"]
HP_LABELS = ["Units 1", "Units 2", "Dropout", "Learning rate", "Batch size"]
MODELS    = ["MLP", "LSTM", "GRU"]
COLORS    = ["#4C72B0", "#DD8452", "#55A868"]

# ── Permutation importance per model ─────────────────────────────────────────
importances = {}
for model in MODELS:
    sub = df[df["model"] == model].copy()
    # encode batch_size (categorical) as numeric
    le = LabelEncoder()
    sub["hp_batch_size"] = le.fit_transform(sub["hp_batch_size"].astype(str))
    X = sub[HP_COLS].values
    y = sub["best_val_mae"].values
    rf = RandomForestRegressor(n_estimators=500, random_state=0)
    rf.fit(X, y)
    pi = permutation_importance(rf, X, y, n_repeats=30, random_state=0)
    importances[model] = pi.importances_mean

# ── Plot ───────────────────────────────────────────────────────────────────────
x      = np.arange(len(HP_LABELS))
width  = 0.25
fig, ax = plt.subplots(figsize=(8, 3.8))

for i, (model, color) in enumerate(zip(MODELS, COLORS)):
    imp = importances[model]
    imp_norm = imp / imp.sum()          # normalize to fractions
    ax.bar(x + (i - 1) * width, imp_norm, width,
           label=model, color=color, edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(HP_LABELS, fontsize=10)
ax.set_ylabel("Relative importance", fontsize=10)
ax.set_ylim(0, None)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(fontsize=9, frameon=False)
ax.grid(axis="y", lw=0.4, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = "/tmp/f7_hp_importance.pdf"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
