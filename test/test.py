import gurobipy as gp
import numpy as np
from gurobipy import GRB

from udgp import M1, M2, Instance, M2Gurobipy

instance = Instance.artificial_molecule(5, freq=True, seed=1234567)

model = M2(instance, nx=4, ny=1)

model.optimize(log=True)

# print(model.Runtime)
