import matplotlib.pyplot as plt
import numpy as np

import udgp

# --- configuração de log --------------------------------------------------
udgp.set_logger("DEBUG", log_to_console=False, log_file="test/teste.log")

# --- parâmetros do experimento -------------------------------------------

N = 6

instance = udgp.Instance.artificial_molecule(N, seed=12345)
solved = instance.solve("M3", backend="gurobipy")

print(instance.work)

works = []

seeds = [1234, 12345, 123456]

mu_values = np.geomspace(1e-3, 1e1, num=10)

# # --- varredura em μ -------------------------------------------------------
# for mu in mu_values:
#     params = {"Mu": mu}

#     work = []
#     for seed in seeds:
#         instance = udgp.Instance.artificial_molecule(N, seed=seed)
#         solved = instance.solve("M3", backend="gurobipy", model_params=params)

#         print(solved)

#         if solved:
#             work.append(instance.work)
#     if work:
#         works.append(np.array(work).mean())
#         print(f"μ = {mu:8.4g}  |  work = {instance.work:10.4f}")
#     else:
#         print(f"μ = {mu:8.4g}  |  Erro!")

# # --- plot log-log e salvamento --------------------------------------------
# plt.figure()
# plt.loglog(mu_values, works, marker="o", linestyle="-")
# plt.xlabel("μ (parâmetro de convexificação)")
# plt.ylabel("Trabalho acumulado (GRB Work Units)")
# plt.title(f"uDGP: instância N={N}")
# plt.grid(True, which="both", ls="--", alpha=0.5)
# plt.tight_layout()
# plt.savefig("test/work_vs_mu.png", dpi=300)
# print("Gráfico salvo em work_vs_mu.png")
