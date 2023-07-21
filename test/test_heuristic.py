import gurobipy as gp
import numpy as np
from gurobipy import GRB
from rich.progress import track

from udgp import M2, M4, generate_random_instance

N = 5

env = gp.Env()

instance = generate_random_instance(N)
model = M2(instance, max_gap=1e-4, log=True, env=env)

model.optimize()

print(model.instance.is_isomorphic())
