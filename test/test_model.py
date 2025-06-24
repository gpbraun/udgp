import udgp

udgp.set_logger("DEBUG", log_to_console=False, log_file="test/teste.log")

N = 5
instance = udgp.Instance.artificial_molecule(
    N,
    freq=True,
    seed=123456,
)

instance.solve("M3", backend="gurobipy")

print(f"WORK: {instance.work}")

print(instance.points)
