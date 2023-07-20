import gurobipy as gp
import numpy as np

from udgp import generate_random_instance, solve_M4

N = 5

instance = generate_random_instance(5)

solve_M4(instance)
