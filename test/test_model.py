import udgp

udgp.set_logger("DEBUG", log_to_console=False, log_file="test/teste.log")

N = 6
instance = udgp.Instance.molecule_artificial(N, seed=12345)

solved = instance.solve("M3", backend="gurobipy")

print(instance.work)
