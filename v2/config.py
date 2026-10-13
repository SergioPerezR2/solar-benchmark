"""
Configuración única del benchmark v2.

Todo lo que define el experimento vive aquí para que cada resultado JSON
pueda guardar una copia exacta de la configuración con la que se produjo.
"""
import os

V2_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(V2_DIR)
RAW_CSV  = os.path.join(BASE_DIR, 'data', 'pvgis_raw.csv')
OUT_DIR  = os.path.join(V2_DIR, 'outputs')
DATA_DIR = os.path.join(OUT_DIR, 'data')
RES_DIR  = os.path.join(OUT_DIR, 'results')
LOG_DIR  = os.path.join(OUT_DIR, 'logs')

# ── Sitio ──────────────────────────────────────────────────────────────────────
LATITUDE   = 20.563425
LONGITUDE  = -100.36943889
ALTITUDE_M = 1820
UTC_OFFSET_H = -6          # CST fijo (sin horario de verano)

# ── Datos ──────────────────────────────────────────────────────────────────────
WINDOW   = 24              # horas de historia
HORIZON  = 1               # t+1
# Variables por paso de la ventana. ghi_cs_next = GHI de cielo despejado de la
# hora siguiente a cada paso (conocida de antemano por geometría solar); en el
# último paso es la del instante a pronosticar.
FEATURES = ['GHI', 'kcs', 'TempAir', 'WindSpeed', 'hour', 'dayofyear', 'month',
            'ghi_cs_next']
DAY_THRESHOLD_GHI = 50.0   # W/m², definición diurna principal (GHI > 50)
DAY_MIN_ELEVATION = 5.0    # grados, definición diurna alternativa
KCS_MIN_ELEVATION = 5.0    # por debajo, kcs se fija en 1 (cielo despejado)
KCS_CLIP = (0.0, 1.5)

# Split principal por años calendario completos (hora local)
SPLIT_YEARS = {'train': [2019, 2020], 'val': [2021], 'test': [2022]}

# Rolling-origin sobre 2022: cada trimestre es un fold de prueba; los 3 meses
# previos sirven para early stopping y todo lo anterior es entrenamiento.
ROLLING_FOLDS = [('2022-01-01', '2022-04-01'), ('2022-04-01', '2022-07-01'),
                 ('2022-07-01', '2022-10-01'), ('2022-10-01', '2023-01-01')]
ROLLING_ES_MONTHS = 3

# ── Espacio de búsqueda (codificación única u ∈ [0,1]^5 para TODOS) ────────────
BATCH_CHOICES = [32, 64, 128]
SPACE = {
    'units_1':       (32, 256),
    'units_2':       (16, 128),
    'dropout_rate':  (0.0, 0.5),
    'log10_lr':      (-4.0, -2.0),
}
DIM = 5
DEFAULT_PARAMS = {'units_1': 64, 'units_2': 32, 'dropout_rate': 0.2,
                  'learning_rate': 1e-3, 'batch_size': 32}

# ── Entrenamiento ──────────────────────────────────────────────────────────────
BUDGET         = 100       # evaluaciones por corrida
OPT_EPOCHS     = 10
OPT_PATIENCE   = 3
FINAL_EPOCHS   = 100
FINAL_PATIENCE = 10
LOW_BUDGETS    = [10, 25, 50]   # incumbentes que también se reentrenan y evalúan

# ── Diseño experimental ────────────────────────────────────────────────────────
MODELS   = ['MLP', 'LSTM', 'GRU']
SEEDS    = list(range(30))
POP_SIZE = 10
CORE_OPTIMIZERS  = ['PSO', 'GA', 'ABC', 'TPE', 'RS']
EXTRA_OPTIMIZERS = ['CMAES', 'BOHB']
POP_SENS_OPTIMIZERS = ['PSO', 'GA', 'ABC']
POP_SENS_SIZES      = [5, 20]
POP_SENS_MODELS     = ['MLP']

# ── Ejecución ──────────────────────────────────────────────────────────────────
N_WORKERS       = int(os.environ.get('BENCH_WORKERS', 6))
MAX_RETRIES     = 2
JOB_TIMEOUT_S   = 8 * 3600
NTFY_TOPIC      = os.environ.get('NTFY_TOPIC', '')   # opcional: tema de ntfy.sh para avisos de avance

# ── Análisis (pre-especificado el 2026-10-07, antes de tener los resultados) ──
TOST_DELTA        = 1.0          # W/m², margen primario de equivalencia
TOST_DELTA_SENS   = [0.5, 2.0]   # sensibilidad
TOST_ALPHA        = 0.05         # => IC del 90 % para TOST
ALPHA             = 0.05
PAIRWISE_TEST     = 'wilcoxon_signed_rank_holm'
