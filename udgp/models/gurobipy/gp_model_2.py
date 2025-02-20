"""Gabriel Braun, 2023
Este módulo implementa o modelo M1 para instâncias do problema uDGP.
"""

import gurobipy as gp
import numpy as np

from .gp_base_model import GPBaseModel


class M2gp(GPBaseModel):
    """Modelo M2 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(M2gp, self).__init__(*args, **kwargs)

        # VARIÁVEIS
        ## Erro no cálculo da distância
        self.p = self.addVars(
            self.IJK,
            name="p",
            vtype=gp.GRB.CONTINUOUS,
            lb=-self.max_gap,
            ub=self.max_gap,
        )
        ## Valor absoluto no erro do cálculo da distância
        self.w = self.addVars(
            self.IJK,
            name="w",
            vtype=gp.GRB.CONTINUOUS,
            lb=0,
            ub=self.max_gap,
        )
        ## Distância k se ela é referente ao par de átomos i e j. 0 em caso contrátrio.
        self.z = self.addVars(
            self.IJK,
            name="z",
            # vtype=gp.GRB.SEMICONT,
            vtype=gp.GRB.CONTINUOUS,
            # lb=self.d_min,
            ub=self.d_max,
        )

        # RESTRIÇÕES
        self.addConstrs(self.w[i, j, k] >= self.p[i, j, k] for i, j, k in self.IJK)
        self.addConstrs(self.w[i, j, k] >= -self.p[i, j, k] for i, j, k in self.IJK)
        self.addConstrs(
            self.z[i, j, k] >= -(1 - self.a[i, j, k]) * self.d_max + self.r[i, j]
            for i, j, k in self.IJK
        )
        self.addConstrs(
            self.z[i, j, k] <= (1 - self.a[i, j, k]) * self.d_max + self.r[i, j]
            for i, j, k in self.IJK
        )
        self.addConstrs(
            self.z[i, j, k] >= -self.a[i, j, k] * self.d_max for i, j, k in self.IJK
        )
        self.addConstrs(
            self.z[i, j, k] <= self.a[i, j, k] * self.d_max for i, j, k in self.IJK
        )
        self.addConstrs(
            self.z[i, j, k]
            >= -(1 - self.a[i, j, k]) * self.d_max + self.dists[k] + self.p[i, j, k]
            for i, j, k in self.IJK
        )
        self.addConstrs(
            self.z[i, j, k]
            <= (1 - self.a[i, j, k]) * self.d_max + self.dists[k] + self.p[i, j, k]
            for i, j, k in self.IJK
        )

        # OBJETIVO
        if self.relaxed:
            self.setObjective(
                1
                + self.w.sum()
                + len(self.IJ)
                - gp.quicksum(
                    self.a[i, j, k] * self.a[i, j, k] for i, j, k in self.IJK
                ),
                gp.GRB.MINIMIZE,
            )
        else:
            self.setObjective(1 + self.w.sum(), gp.GRB.MINIMIZE)
