import gurobipy as gp
import numpy as np
from gurobipy import GRB
from rich.progress import track

from udgp import M2, M4, generate_random_instance

N = 5

instance_num = 100

env = gp.Env()

for gap in [6.0e-4, 1.29e-5]:
    if gap == 0.1:
        continue

    print(f"max_gap = {gap:e}")
    t = np.empty(instance_num, dtype="float16")
    s = np.empty(instance_num, dtype="bool")

    for i in track(range(instance_num)):
        instance = generate_random_instance(N)
        model = M4(instance, max_gap=gap, log=False, env=env)

        model.optimize()

        if model.Status == GRB.OPTIMAL:
            t[i] = model.Runtime
            s[i] = model.instance.is_isomorphic()

    print("=" * 50)
    print(f"max_gap = {gap:e}")
    print(f"t_mean = {np.mean(t)}")
    print(f"t_std = {np.std(t)}")
    print(f"s_mean = {np.mean(s)}")
    print(f"s_std = {np.std(s)}")
    print("=" * 50)
