"""Gabriel Braun, 2023

Este módulo implementa o modelo M4 para instâncias do problema uDGP.
"""

import gurobipy as gp
from gurobipy import GRB

from .base_model import BaseModel


class M4(BaseModel):
    """Modelo M4 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(M4, self).__init__(*args, **kwargs)

        # VARIÁVEIS
        ## Erro no cálculo da distância
        self.p = self.addVars(
            self.ijk_indices(),
            name="p",
            vtype=GRB.CONTINUOUS,
            lb=-self.max_gap,
            ub=self.max_gap,
        )
        ## Valor absoluto no erro do cálculo da distância
        self.w = self.addVars(
            self.ijk_indices(),
            name="w",
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=self.max_gap,
        )
        ## Distância k se ela é referente ao par de átomos i e j. 0 em caso contrátrio.
        self.z = self.addVars(
            self.ijk_indices(),
            name="z",
            vtype=GRB.SEMICONT,
            lb=0.5,
            ub=self.d_max,
        )

        # RESTRIÇÕES
        distances = self.instance.dists
        self.addConstrs(
            self.w[i, j, k] >= self.p[i, j, k] for i, j, k in self.ijk_indices()
        )
        self.addConstrs(
            self.w[i, j, k] >= -self.p[i, j, k] for i, j, k in self.ijk_indices()
        )
        self.addConstrs(
            self.z[i, j, k] >= -(1 - self.a[i, j, k]) * self.d_max + self.r[i, j]
            for i, j, k in self.ijk_indices()
        )
        self.addConstrs(
            self.z[i, j, k] <= (1 - self.a[i, j, k]) * self.d_max + self.r[i, j]
            for i, j, k in self.ijk_indices()
        )
        self.addConstrs(
            self.z[i, j, k] >= -self.a[i, j, k] * self.d_max
            for i, j, k in self.ijk_indices()
        )
        self.addConstrs(
            self.z[i, j, k] <= self.a[i, j, k] * self.d_max
            for i, j, k in self.ijk_indices()
        )
        self.addConstrs(
            self.z[i, j, k]
            >= -(1 - self.a[i, j, k]) * self.d_max + (distances[k] + self.p[i, j, k])
            for i, j, k in self.ijk_indices()
        )
        self.addConstrs(
            self.z[i, j, k]
            <= (1 - self.a[i, j, k]) * self.d_max + (distances[k] + self.p[i, j, k])
            for i, j, k in self.ijk_indices()
        )

        # OBJETIVO
        if self.relaxed:
            self.setObjective(
                1
                + self.w.sum()
                + len(list(self.ij_indices()))
                - gp.quicksum(
                    self.a[i, j, k] * self.a[i, j, k] for i, j, k in self.ijk_indices()
                ),
                GRB.MINIMIZE,
            )
        else:
            self.setObjective(1 + self.w.sum(), GRB.MINIMIZE)

        self.update()
