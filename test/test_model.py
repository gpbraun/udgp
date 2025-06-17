from udgp import Instance

N = 5
instance = Instance.artificial_molecule(
    N,
    freq=True,
    seed=123456,
)

instance.solve("M2", log=True, backend="gurobipy")

print(instance.runtime)
