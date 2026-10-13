"""
I2 — Annual GHI climatology: heatmap (hour × month) + seasonal diurnal profiles
Reads: ../data/pvgis_raw.csv
Saves: ../Figures/i2_ghi_climate.{pdf,png}
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "pvgis_raw.csv"
OUT  = Path(__file__).resolve().parents[3] / "Figures"
if not OUT.is_dir():                       # fuera de la carpeta del manuscrito
    OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["TIMESTAMP"], index_col="TIMESTAMP")
# v2: hora local fija UTC-6 (sin horario de verano)
df.index = df.index.tz_convert(None) + pd.Timedelta(hours=-6)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
SEASONS = {
    "Winter (Dec–Feb)": [12, 1, 2],
    "Spring (Mar–May)": [3, 4, 5],
    "Summer (Jun–Aug)": [6, 7, 8],
    "Autumn (Sep–Nov)": [9, 10, 11],
}
SEASON_COLORS = ["#0077BB", "#EE7733", "#CC3311", "#009988"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, (ax_hm, ax_pr) = plt.subplots(1, 2, figsize=(7.0, 3.4))
fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.14, wspace=0.62)

# ── Heatmap: mean GHI by hour × month ────────────────────────────────────────
pivot = (df.groupby([df.index.month, df.index.hour])["GHI"]
           .mean()
           .unstack(level=0))          # rows = hour, cols = month

im = ax_hm.imshow(pivot.values, aspect="auto", origin="lower",
                  cmap="YlOrRd",
                  norm=mcolors.Normalize(vmin=0, vmax=pivot.values.max()))

ax_hm.set_xticks(range(12))
ax_hm.set_xticklabels(MONTH_NAMES, fontsize=7.8, rotation=45, ha="right")
ax_hm.set_yticks(range(0, 24, 3))
ax_hm.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=7.5)
ax_hm.set_xlabel("Month", fontsize=9)
ax_hm.set_ylabel("Hour of day (CST)", fontsize=9)
ax_hm.set_title("(a) Mean GHI (W/m²)", fontsize=9, fontweight="bold", pad=4)
ax_hm.tick_params(length=0)

cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=7.5)
cbar.set_label("W/m²", fontsize=8)

# ── Seasonal diurnal profiles ─────────────────────────────────────────────────
for (label, months), color in zip(SEASONS.items(), SEASON_COLORS):
    mask    = df.index.month.isin(months)
    profile = df[mask].groupby(df[mask].index.hour)["GHI"].agg(["mean", "std"])
    hrs     = profile.index.values
    mu      = profile["mean"].values
    sigma   = profile["std"].values
    ax_pr.plot(hrs, mu, color=color, lw=1.4, label=label)
    ax_pr.fill_between(hrs, np.maximum(mu - sigma, 0), mu + sigma,
                       color=color, alpha=0.15)

ax_pr.set_xticks(range(0, 24, 3))
ax_pr.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)],
                      fontsize=7.5, rotation=35, ha="right")
ax_pr.set_xlabel("Hour of day (CST)", fontsize=9)
ax_pr.set_ylabel("Mean GHI ± std (W/m²)", fontsize=9, labelpad=2)
ax_pr.set_ylim(0, 1380)
ax_pr.set_title("(b) Seasonal diurnal profile", fontsize=9, fontweight="bold", pad=4)
ax_pr.set_xlim(0, 23)
ax_pr.set_ylim(0)
ax_pr.legend(fontsize=6.6, frameon=False, loc="upper left", bbox_to_anchor=(-0.01, 1.0),
             handlelength=1.2, labelspacing=0.3)
ax_pr.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"i2_ghi_climate.{ext}", dpi=300, bbox_inches="tight")
print(f"Saved {OUT}/i2_ghi_climate.pdf/.png")
