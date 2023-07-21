"""Gabriel Braun, 2023

Este módulo implementa o modelo M1 para instâncias do problema uDGP.
"""

from gurobipy import GRB

from .base_model import BaseModel


class M1(BaseModel):
    """Modelo M1 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(M1, self).__init__(*args, **kwargs)

        # VARIÁVEIS
        self.s = self.addVars(
            self.ijk_values(),
            name="s",
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
        )
        self.t = self.addVars(
            self.ijk_values(),
            name="t",
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
        )
        self.u = self.addVars(
            self.ijk_values(),
            name="u",
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
        )

        # RESTRIÇÕES
        distances = self.instance.distances
        self.addConstrs(
            self.s[i, j, k] == self.r[i, j] * self.r[i, j] - distances[k] * distances[k]
            for i, j, k in self.ijk_values()
        )
        self.addConstrs(
            self.t[i, j, k] == self.s[i, j, k] * self.s[i, j, k]
            for i, j, k in self.ijk_values()
        )
        self.addConstrs(
            self.u[i, j, k] == self.a[i, j, k] * self.t[i, j, k]
            for i, j, k in self.ijk_values()
        )

        # OBJETIVO
        self.setObjective(self.u.sum(), GRB.MINIMIZE)

        self.update()
