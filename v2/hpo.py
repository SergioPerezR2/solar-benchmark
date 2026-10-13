"""
Optimizadores de hiperparámetros sobre una codificación ÚNICA u ∈ [0,1]^5.

Todos los métodos (PSO, GA, ABC, RS, TPE, CMA-ES, BOHB) proponen vectores en el
mismo hipercubo y se decodifican con `decode`, de modo que ninguno tiene
ventaja por la representación del espacio.

Cada optimizador recibe un `Evaluator` que cuenta el presupuesto; cuando se
agota, `evaluate` lanza `BudgetExhausted` y el optimizador termina.
"""
import numpy as np
import optuna

import config as C

LOWER = np.zeros(C.DIM)
UPPER = np.ones(C.DIM)


def decode(u):
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    lo1, hi1 = C.SPACE['units_1']
    lo2, hi2 = C.SPACE['units_2']
    lod, hid = C.SPACE['dropout_rate']
    lol, hil = C.SPACE['log10_lr']
    b = min(int(np.floor(u[4] * len(C.BATCH_CHOICES))), len(C.BATCH_CHOICES) - 1)
    return {
        'units_1':       int(round(lo1 + u[0] * (hi1 - lo1))),
        'units_2':       int(round(lo2 + u[1] * (hi2 - lo2))),
        'dropout_rate':  float(lod + u[2] * (hid - lod)),
        'learning_rate': float(10 ** (lol + u[3] * (hil - lol))),
        'batch_size':    C.BATCH_CHOICES[b],
    }


class BudgetExhausted(Exception):
    pass


class Evaluator:
    """Envuelve la función objetivo y lleva el conteo del presupuesto.

    fn(u, report) -> (mae, epochs, pruned). El presupuesto es de evaluaciones
    (`budget`) y, opcionalmente, de épocas totales (`epoch_budget`, usado por
    BOHB para igualar el cómputo nominal B x E_opt de los demás métodos).
    """

    def __init__(self, fn, budget, epoch_budget=None):
        self.fn = fn
        self.budget = budget
        self.epoch_budget = epoch_budget
        self.n = 0
        self.epochs = 0

    def remaining(self):
        if self.epoch_budget is not None and self.epochs >= self.epoch_budget:
            return 0
        return self.budget - self.n

    def evaluate(self, u, report=None):
        if self.remaining() <= 0:
            raise BudgetExhausted
        self.n += 1
        mae, epochs, pruned = self.fn(np.clip(u, 0, 1), report)
        self.epochs += epochs
        return (mae, pruned) if report is not None else mae


# ── Random Search ──────────────────────────────────────────────────────────────
def run_rs(ev, seed, pop=None):
    rng = np.random.default_rng(seed)
    try:
        while True:
            ev.evaluate(rng.random(C.DIM))
    except BudgetExhausted:
        pass


# ── PSO ────────────────────────────────────────────────────────────────────────
def run_pso(ev, seed, pop=C.POP_SIZE, w=0.7, c1=1.5, c2=1.5):
    rng = np.random.default_rng(seed)
    pos = rng.random((pop, C.DIM))
    vel = np.zeros((pop, C.DIM))
    try:
        pbest_fit = np.array([ev.evaluate(p) for p in pos])
        pbest_pos = pos.copy()
        g = int(np.argmin(pbest_fit))
        gbest_pos, gbest_fit = pbest_pos[g].copy(), pbest_fit[g]
        while True:
            r1 = rng.random((pop, C.DIM))
            r2 = rng.random((pop, C.DIM))
            vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
            pos = np.clip(pos + vel, LOWER, UPPER)
            for i in range(pop):
                f = ev.evaluate(pos[i])
                if f < pbest_fit[i]:
                    pbest_fit[i], pbest_pos[i] = f, pos[i].copy()
                    if f < gbest_fit:
                        gbest_fit, gbest_pos = f, pos[i].copy()
    except BudgetExhausted:
        pass


# ── GA ─────────────────────────────────────────────────────────────────────────
def run_ga(ev, seed, pop=C.POP_SIZE, mutation_rate=0.2, elite_size=2, tour=3):
    """GA real con elitismo. Las élites conservan su aptitud y no se
    re-evalúan, así que todo el presupuesto se usa en individuos nuevos."""
    rng = np.random.default_rng(seed)
    P = rng.random((pop, C.DIM))
    fit = np.full(pop, np.inf)
    try:
        for i in range(pop):
            fit[i] = ev.evaluate(P[i])
        while True:
            order = np.argsort(fit)
            new_P = [P[i].copy() for i in order[:elite_size]]
            new_fit = [fit[i] for i in order[:elite_size]]

            def tournament():
                idx = rng.choice(pop, min(tour, pop), replace=False)
                return P[idx[np.argmin(fit[idx])]]

            while len(new_P) < pop:
                p1, p2 = tournament(), tournament()
                a = rng.random(C.DIM)
                child = a * p1 + (1 - a) * p2
                mask = rng.random(C.DIM) < mutation_rate
                child[mask] += rng.normal(0, 0.1, mask.sum())
                child = np.clip(child, LOWER, UPPER)
                new_fit.append(ev.evaluate(child))
                new_P.append(child)
            P, fit = np.array(new_P), np.array(new_fit)
    except BudgetExhausted:
        pass


# ── ABC ────────────────────────────────────────────────────────────────────────
def run_abc(ev, seed, pop=C.POP_SIZE, limit=15):
    rng = np.random.default_rng(seed)
    n_emp = max(2, pop // 2)
    S = rng.random((n_emp, C.DIM))
    trials = np.zeros(n_emp)
    try:
        fit = np.array([ev.evaluate(s) for s in S])

        def neighbour(i):
            k = rng.choice([j for j in range(n_emp) if j != i])
            phi = rng.uniform(-1, 1, C.DIM)
            return np.clip(S[i] + phi * (S[i] - S[k]), LOWER, UPPER)

        def greedy(i, cand):
            f = ev.evaluate(cand)
            if f < fit[i]:
                S[i], fit[i], trials[i] = cand, f, 0
            else:
                trials[i] += 1

        while True:
            for i in range(n_emp):
                greedy(i, neighbour(i))
            q = 1.0 / (fit + 1e-9)
            q /= q.sum()
            for _ in range(n_emp):
                i = rng.choice(n_emp, p=q)
                greedy(i, neighbour(i))
            for i in range(n_emp):
                if trials[i] > limit:
                    S[i] = rng.random(C.DIM)
                    fit[i] = ev.evaluate(S[i])
                    trials[i] = 0
    except BudgetExhausted:
        pass


# ── Optuna: TPE, CMA-ES y BOHB (TPE + Hyperband) ──────────────────────────────
def _optuna(ev, sampler, pruner=None):
    study = optuna.create_study(direction='minimize', sampler=sampler,
                                pruner=pruner or optuna.pruners.NopPruner())

    def objective(trial):
        u = np.array([trial.suggest_float(f'u{i}', 0.0, 1.0) for i in range(C.DIM)])
        if pruner is None:
            return ev.evaluate(u)

        def report(epoch, value):
            trial.report(value, epoch)
            return trial.should_prune()

        mae, pruned = ev.evaluate(u, report)
        if pruned:
            raise optuna.TrialPruned()
        return mae

    while ev.remaining() > 0:
        try:
            study.optimize(objective, n_trials=1, catch=())
        except BudgetExhausted:
            break


def run_tpe(ev, seed, pop=None):
    _optuna(ev, optuna.samplers.TPESampler(seed=seed))


def run_cmaes(ev, seed, pop=None):
    _optuna(ev, optuna.samplers.CmaEsSampler(seed=seed, n_startup_trials=0,
                                             popsize=pop))


def run_bohb(ev, seed, pop=None):
    # Hyperband sobre épocas (1..OPT_EPOCHS, η=3) con muestreo TPE. El
    # presupuesto es de épocas (ver Evaluator.epoch_budget): al podar trials
    # malos, BOHB puede probar más de B configuraciones con el mismo cómputo.
    _optuna(ev, optuna.samplers.TPESampler(seed=seed),
            optuna.pruners.HyperbandPruner(min_resource=1, max_resource=C.OPT_EPOCHS,
                                           reduction_factor=3))


OPTIMIZERS = {'RS': run_rs, 'PSO': run_pso, 'GA': run_ga, 'ABC': run_abc,
              'TPE': run_tpe, 'CMAES': run_cmaes, 'BOHB': run_bohb}
