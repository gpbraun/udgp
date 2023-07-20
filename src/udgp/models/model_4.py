import gurobipy as gp
from gurobipy import GRB

from .base_model import BaseModel


def solve_M4(instance):
    """Cria o modelo M4 no Gurobi para o problema da distâncias não associadas."""
    model = BaseModel(instance)

    model.setParam("SolutionLimit", 1)

    D = model.distances.max()

    # VARIÁVEIS
    ## Erro no cálculo da distância
    p = model.addMVar(model.m, name="p", vtype=GRB.CONTINUOUS, lb=-0.002, ub=+0.002)
    ## Valor absoluto no erro do cálculo da distância
    w = model.addMVar(model.m, name="w", vtype=GRB.CONTINUOUS, ub=0.002)
    ## Distância k se ela é referente ao par de átomos i e j e 0 em caso contrátrio.
    z = model.addVars(model.ijk_values(), name="z", vtype=GRB.CONTINUOUS)

    # OBJETIVO
    model.setObjective(gp.quicksum(w), GRB.MINIMIZE)

    # RESTRIÇÕES
    model.addConstrs(w[k] >= +p[k] for k in model.k_values())
    model.addConstrs(w[k] >= -p[k] for k in model.k_values())
    model.addConstrs(
        -(1 - model.a[i, j, k]) * D + model.r[i, j] <= z[i, j, k]
        for i, j, k in model.ijk_values()
    )
    model.addConstrs(
        z[i, j, k] <= (1 - model.a[i, j, k]) * D + model.r[i, j]
        for i, j, k in model.ijk_values()
    )
    model.addConstrs(
        -model.a[i, j, k] * D <= z[i, j, k] for i, j, k in model.ijk_values()
    )
    model.addConstrs(
        z[i, j, k] <= model.a[i, j, k] * D for i, j, k in model.ijk_values()
    )
    model.addConstrs(
        -(1 - model.a[i, j, k]) * D + (model.distances[k] + p[k]) <= z[i, j, k]
        for i, j, k in model.ijk_values()
    )
    model.addConstrs(
        z[i, j, k] <= (1 - model.a[i, j, k]) * D + (model.distances[k] + p[k])
        for i, j, k in model.ijk_values()
    )

    # model.relax()

    print(f"Resolvendo modelo...")

    # model.update()
    # model.write('model.mps')

    model.optimize()

    print(f"Solução encontrada em: {model.Runtime:.2f} s")

    for k in range(model.m):
        for i in range(model.n - 1):
            for j in range(i + 1, model.n):
                if model.a[i, j, k].X > 0.1:
                    print(
                        f"k = {k+1} -> ({i+1},{j+1}). ({model.distances[k]} -> {model.r[i, j].X})"
                    )

    print(model.x.X)

    return model.x.X, model.Runtime
