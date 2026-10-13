"""
Ejemplo: evaluar un optimizador que NO está en el artículo sobre el benchmark sustituto.

Se usa Evolución Diferencial (scipy.optimize.differential_evolution), una
metaheurística frecuente en la literatura de pronóstico solar, y se compara con los
seis optimizadores del artículo bajo el mismo protocolo (30 semillas pareadas,
100 evaluaciones, equivalencia con margen de 1 W/m²).

    cd v2 && python examples/try_new_optimizer.py
"""
import os
import sys

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import surrogate_benchmark as S  # noqa: E402


def my_optimizer(objective, budget, seed):
    """Cualquier función con esta firma sirve: recibe objective(u) -> MAE de
    validación, con u en [0, 1]^5, el presupuesto y una semilla."""
    differential_evolution(objective, bounds=[(0, 1)] * 5, popsize=2, maxiter=10_000,
                           init='random', polish=False, tol=0, seed=seed)


if __name__ == '__main__':
    summary, comparison = S.compare_new_optimizer(my_optimizer, name='DE')
    print('Mean best validation MAE (W/m2) after b evaluations:')
    print(summary.pivot_table(index=['model', 'optimizer'], columns='eval', values='mean_best').round(2).to_string())
    print('\nDE minus each optimizer of the article at b = 100 (paired by seed):')
    c = comparison[comparison['eval'] == 100]
    print(c[['model', 'vs', 'mean_diff', 'ci90_lo', 'ci90_hi', 'better', 'worse', 'equivalent_d1']]
          .round(2).to_string(index=False))
