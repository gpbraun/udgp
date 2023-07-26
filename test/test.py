import gurobipy as gp
import numpy as np
from gurobipy import GRB

from udgp import M2, M4, Instance

env = gp.Env()

instance = Instance.random(5, freq=True)
model = M4(instance, env=env, max_gap=1e-2)
# model.optimize(log=True)
