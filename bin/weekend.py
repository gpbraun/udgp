import gurobipy as gp
import numpy as np
from gurobipy import GRB
from rich.progress import track

from udgp import M1, M2, M4, M5, generate_random_instance

N = 5

instance_num = 100

env = gp.Env()

f = open("bin\\weekend_out.txt", "w")

models = {
    "M4": M4,
    "M5": M5,
    "M2": M2,
    "M1": M1,
}


def newprint(string, *args, **kwargs):
    print(string, flush=True)
    print(string, flush=True, *args, **kwargs)


#################################################
# Tempo para achar um core
#################################################

newprint("=" * 50, file=f)
newprint(f"Tempo para achar um core", file=f)
newprint("=" * 50, file=f)

for name, M in models.items():
    newprint("=" * 50, file=f)
    newprint(name, file=f)
    newprint("=" * 50, file=f)
    for n in [10, 20, 50, 100, 200, 300, 400]:
        newprint(f"n = {n}", file=f)
        t = []
        timeout = []

        for i in track(range(instance_num)):
            instance = generate_random_instance(N)
            model = M(instance, n=5, log=False, env=env)

            model.setParam("TimeLimit", 60 * 60)

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                t.append(model.Runtime)

            if model.Status == GRB.TIME_LIMIT:
                timeout.append(True)
            else:
                timeout.append(False)

        newprint("-" * 50, file=f)
        newprint(f"n = {n}", file=f)
        newprint(f"t_mean = {np.mean(t)}", file=f)
        newprint(f"t_std = {np.std(t)}", file=f)
        newprint(f"timeouts = {100 * np.mean(timeout)}%", file=f)
        newprint("-" * 50, file=f)

#################################################
# MIPGap e a qualidade e tempo da otimização
#################################################

newprint("=" * 50, file=f)
newprint(f"MIPGap e a qualidade e tempo da otimização", file=f)
newprint("=" * 50, file=f)

for name, M in models.items():
    newprint("=" * 50, file=f)
    newprint(name, file=f)
    newprint("=" * 50, file=f)
    for gap in np.logspace(-1, -5, 10, endpoint=True, base=10):
        newprint(f"max_gap = {gap:e}", file=f)
        t = []
        s = []
        timeout = []

        for i in track(range(instance_num)):
            instance = generate_random_instance(N)
            model = M(instance, max_gap=gap, log=False, env=env)

            model.setParam("TimeLimit", 10 * 60)

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                t.append(model.Runtime)
                s.append(model.instance.is_isomorphic())

            if model.Status == GRB.TIME_LIMIT:
                timeout.append(True)
            else:
                timeout.append(False)

        newprint("-" * 50, file=f)
        newprint(f"max_gap = {gap:e}", file=f)
        newprint(f"t_mean = {np.mean(t)}", file=f)
        newprint(f"t_std = {np.std(t)}", file=f)
        newprint(f"hits = {100 * np.mean(s)}%", file=f)
        newprint(f"timeouts = {100 * np.mean(timeout)}%", file=f)
        newprint("-" * 50, file=f)

f.close()
