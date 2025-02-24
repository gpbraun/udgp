import gurobipy as gp
import numpy as np

from udgp import Instance

print(gp.__version__)


N = 6
instance = Instance.artificial_molecule(
    N,
    freq=True,
    seed=123456,
)
# instance.view_input()

instance.reset(reset_runtime=True)
instance.solve_step("M3", time_limit=300, log=True)

print(instance.points)

print(instance.runtime)
