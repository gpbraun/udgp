import udgp

udgp.set_logger("DEBUG", log_to_console=False, log_file="test/teste.log")

N = 5
instance = udgp.Instance.molecule_artificial(N, seed=123456)

solved = instance.solve(
    "M2",
    backend="gurobipy",
    model_params={"Relax": 0},
)

print(instance.work)
