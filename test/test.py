import gurobipy as gp
import numpy as np
from gurobipy import GRB

from udgp import M2, M4, generate_random_instance

env = gp.Env()

N = 6

instance = generate_random_instance(N)
model = M4(instance, env=env)

model.optimize(log=True)

# if model.Status == GRB.OPTIMAL:
print(f"Tempo: {model.Runtime} s")
print(f"Isomorfo: {model.instance.is_isomorphic()}")
