import numpy as np

from udgp import Instance

N = 5
instance = Instance.artificial_molecule(N, freq=True, seed=123456)
# instance.view_input()

instance.reset(reset_runtime=True)
instance.solve_step("M3GP", max_gap=1e-4, time_limit=300, log=True)

print(instance.points)

print(instance.runtime)
