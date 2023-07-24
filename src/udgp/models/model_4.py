"""Gabriel Braun, 2023

Este módulo implementa o modelo M4 para instâncias do problema uDGP.
"""

from gurobipy import GRB

from .base_model import BaseModel


class M4(BaseModel):
    """Modelo M4 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(M4, self).__init__(*args, **kwargs)

        # VARIÁVEIS
        ## Erro no cálculo da distância
        self.p = self.addVars(
            self.m,
            name="p",
            vtype=GRB.CONTINUOUS,
            lb=-self.Params.MIPGap,
            ub=self.Params.MIPGap,
        )
        ## Valor absoluto no erro do cálculo da distância
        self.w = self.addVars(
            self.m,
            name="w",
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=self.Params.MIPGap,
        )
        ## Distância k se ela é referente ao par de átomos i e j. 0 em caso contrátrio.
        self.z = self.addVars(
            self.ijk_values(),
            name="z",
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=GRB.INFINITY,
        )

        # RESTRIÇÕES
        distances = self.instance.distances
        D = distances.max()
        self.addConstrs(self.w[k] >= self.p[k] for k in self.k_values())
        self.addConstrs(self.w[k] >= -self.p[k] for k in self.k_values())
        self.addConstrs(
            -(1 - self.a[i, j, k]) * D + self.r[i, j] <= self.z[i, j, k]
            for i, j, k in self.ijk_values()
        )
        self.addConstrs(
            self.z[i, j, k] <= (1 - self.a[i, j, k]) * D + self.r[i, j]
            for i, j, k in self.ijk_values()
        )
        self.addConstrs(
            self.z[i, j, k] >= -self.a[i, j, k] * D for i, j, k in self.ijk_values()
        )
        self.addConstrs(
            self.z[i, j, k] <= self.a[i, j, k] * D for i, j, k in self.ijk_values()
        )
        self.addConstrs(
            self.z[i, j, k] >= -(1 - self.a[i, j, k]) * D + (distances[k] + self.p[k])
            for i, j, k in self.ijk_values()
        )
        self.addConstrs(
            self.z[i, j, k] <= (1 - self.a[i, j, k]) * D + (distances[k] + self.p[k])
            for i, j, k in self.ijk_values()
        )

        # OBJETIVO
        self.setObjective(self.w.sum(), GRB.MINIMIZE)

        self.update()
