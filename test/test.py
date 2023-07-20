import numpy as np
from gurobipy import GRB

from udgp import M4, generate_random_instance

N = 5

instance_num = 100

for gap in [
    1e-1,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
    1e-9,
    1e-10,
]:
    print(gap)
    t = np.empty(instance_num, dtype="float16")
    s = np.empty(instance_num, dtype="bool")

    for i in range(instance_num):
        input_instance = generate_random_instance(N)
        model = M4(input_instance, max_gap=gap, log=True)

        model.optimize()

        if model.Status == GRB.OPTIMAL:
            t[i] = model.Runtime
            s[i] = model.solution_is_isomorphic()

    print("=" * 50)
    print(f"max_gap = {gap}")
    print(f"t_mean = {np.mean(t)}")
    print(f"t_std = {np.std(t)}")
    print(f"s_mean = {np.mean(s)}")
    print(f"s_std = {np.std(s)}")
    print("=" * 50)
