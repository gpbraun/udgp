"""Gabriel Braun, 2023

Este módulo implementa o modelo M1 para instâncias do problema uDGP.
"""

import gurobipy as gp
from gurobipy import GRB

from .base_model import BaseModel


class M1(BaseModel):
    """Modelo M1 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(M1, self).__init__(*args, **kwargs)

        # VARIÁVEIS
        self.s = self.addVars(
            self.ijk_indices(),
            name="s",
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
        )
        self.t = self.addVars(
            self.ijk_indices(),
            name="t",
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
        )
        self.u = self.addVars(
            self.ijk_indices(),
            name="u",
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
        )

        # RESTRIÇÕES
        distances = self.instance.dists
        self.addConstrs(
            self.s[i, j, k] == self.r[i, j] * self.r[i, j] - distances[k] ** 2
            for i, j, k in self.ijk_indices()
        )
        self.addConstrs(
            self.t[i, j, k] == self.s[i, j, k] * self.s[i, j, k]
            for i, j, k in self.ijk_indices()
        )
        self.addConstrs(
            self.u[i, j, k] == self.a[i, j, k] * self.t[i, j, k]
            for i, j, k in self.ijk_indices()
        )

        # OBJETIVO
        if self.relaxed:
            self.setObjective(
                1
                + self.u.sum()
                - gp.quicksum(
                    self.a[i, j, k] * self.a[i, j, k] - 1
                    for i, j, k in self.ijk_indices()
                ),
                GRB.MINIMIZE,
            )
        else:
            self.setObjective(self.u.sum() + 1, GRB.MINIMIZE)

        self.update()
