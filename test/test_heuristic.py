import gurobipy as gp
import numpy as np
from gurobipy import GRB

from udgp import M1, M2, M4, M5, Instance

env = gp.Env()

instance = Instance.lj_cluster(20, freq=True)

previous_a = []

time = 0

i = 1
while not instance.is_solved():
    # while instance.fixed_n < instance.n:
    print("=================")
    print(f"TENTATIVA {i}")
    print("=================")
    i += 1

    # instance.reset()
    # m_core = M4(instance, nx=4, max_gap=5e-2, env=env)
    # for a_indices in previous_a:
    #     m_core.addConstr(
    #         gp.quicksum(m_core.a[i, j, k] for i, j, k in a_indices)
    #         <= len(a_indices) - 1
    #     )
    # m_core.optimize()
    # time += m_core.Runtime
    # previous_a.append(instance.a_indices.tolist())

    instance.reset_with_core("mock")

    print("CORE")

    broken = False
    while instance.fixed_n < instance.n and not broken:
        print(f"  {instance.fixed_n} átomos")

        solved = False
        previous_aa = []
        tries = 0

        while not solved:
            tries += 1

            m = M4(instance, nx=1, max_gap=5e-2, env=env)

            for a_indices in previous_aa:
                try:
                    m.addConstr(
                        gp.quicksum(m.a[i, j, k] for i, j, k in a_indices)
                        <= len(a_indices) - 1
                    )
                except:
                    pass

            solved = m.optimize()
            previous_aa.append(instance.a_indices.tolist())

            time += m.Runtime

            if m.Status == GRB.INFEASIBLE or tries > 200:
                broken = True
                # previous_a.append([(i, j, k) for i, j, k in instance.a_indices])
                break

print(instance.is_solved())
print(time)
