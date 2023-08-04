import gurobipy as gp
import numpy as np
from gurobipy import GRB

from udgp import M1, M2, M4, M5, Instance

env = gp.Env()

instance = Instance.lj_cluster(8, freq=True)
m = M4(
    instance,
    # nx=4,
    max_gap=1e-2,
    env=env,
)
m.optimize(log=False)

print(instance.is_solved())
print(m.Runtime)
