"""
Fig. 1 (v2) — Serie de GHI: semana despejada, semana variable y detalle de 48 h
con la ventana de entrada (t-24 ... t-1) y la hora pronosticada t.

Cambios respecto a v1: hora local UTC-6 fija (sin horario de verano), detalle de
exactamente 48 h y curva de GHI de cielo despejado (Ineichen), que ahora es una
variable de entrada del modelo.

    python figures/i1_ghi_timeseries.py   (desde v2/)
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

_FIG = os.path.join(C.BASE_DIR, '..', 'Figures')          # carpeta del manuscrito, si existe
OUT = _FIG if os.path.isdir(_FIG) else os.path.join(C.OUT_DIR, 'figures')

d = np.load(os.path.join(C.DATA_DIR, 'dataset.npz'))
s = pd.DataFrame({'GHI': d['ghi'], 'GHI_cs': d['ghi_cs']},
                 index=pd.to_datetime(d['time_local'])).sort_index()

clear = s['2021-04-19':'2021-04-25 23:59']
cloud = s['2021-07-19':'2021-07-25 23:59']
z0 = pd.Timestamp('2021-07-21 00:30')
zoom = s[z0:z0 + pd.Timedelta(hours=47)]               # 48 valores horarios
t_pred = pd.Timestamp('2021-07-22 13:30')
t_first = t_pred - pd.Timedelta(hours=24)              # primer paso de la ventana
t_last = t_pred - pd.Timedelta(hours=1)                # último paso de la ventana

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(7.0, 4.4))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28, left=0.09, right=0.97, top=0.93, bottom=0.13)
ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), None, fig.add_subplot(gs[1, :])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

for ax, w, col, title in ((ax1, clear, '#EE7733', '(a) Clear-sky week (Apr 2021)'),
                          (ax2, cloud, '#0077BB', '(b) Variable week (Jul 2021)')):
    ax.plot(w.index, w.GHI_cs, color='0.45', lw=0.8, ls='--', label='Clear-sky GHI')
    ax.fill_between(w.index, w.GHI, alpha=0.25, color=col)
    ax.plot(w.index, w.GHI, color=col, lw=0.9, label='GHI')
    ax.set_title(title, fontsize=8.5, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    ax.set_xlabel('Day', fontsize=8)
    ax.set_ylim(0, 1250)
ax1.set_ylabel('GHI (W/m²)', fontsize=8.5)
ax2.tick_params(labelleft=False)

ax3.axvspan(t_first - pd.Timedelta(minutes=30), t_last + pd.Timedelta(minutes=30),
            alpha=0.12, color='#CC3311', zorder=0)
ax3.plot(zoom.index, zoom.GHI_cs, color='0.45', lw=0.8, ls='--')
ax3.fill_between(zoom.index, zoom.GHI, alpha=0.18, color='#009988')
ax3.plot(zoom.index, zoom.GHI, color='#009988', lw=1.1, marker='o', ms=2)
ax3.plot([t_pred], [s.loc[t_pred, 'GHI']], marker='*', ms=11, color='#CC3311', zorder=4)
ax3.set_title('(c) 48-hour detail: input window and forecast target', fontsize=8.5, fontweight='bold')
ax3.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=25, ha='right', fontsize=7.5)
ax3.set_ylabel('GHI (W/m²)', fontsize=8.5)
ax3.set_xlabel('Local time (CST, UTC$-$6)', fontsize=8)
ax3.set_ylim(0, 1250)

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
fig.legend([Line2D([], [], color='0.25', lw=1.1), Line2D([], [], color='0.45', lw=0.8, ls='--'),
            Patch(facecolor='#CC3311', alpha=0.12), Line2D([], [], color='#CC3311', marker='*', ms=9, ls='none')],
           ['GHI', 'Clear-sky GHI', 'Input window (24 h)', 'Forecast target $t$'],
           loc='lower center', ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.955))
os.makedirs(OUT, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'i1_ghi_timeseries.{ext}'), dpi=300, bbox_inches='tight')
print('Guardado', os.path.abspath(os.path.join(OUT, 'i1_ghi_timeseries.pdf')),
      f'(detalle: {len(zoom)} h, {zoom.index[0]} -> {zoom.index[-1]})')
