"""Gabriel Braun, 2023
Este módulo implementa o modelo M1 para instâncias do problema uDGP.
"""

import gurobipy as gpy

from .gpy_base_model import gpyBaseModel


class gpyM1(gpyBaseModel):
    """Modelo M1 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(gpyM1, self).__init__(*args, **kwargs)
        self.name = "M1"

        # VARIÁVEIS
        self.s = self.addVars(
            self.IJK,
            name="s",
            vtype=gpy.GRB.CONTINUOUS,
            lb=-gpy.GRB.INFINITY,
            ub=gpy.GRB.INFINITY,
        )
        self.t = self.addVars(
            self.IJK,
            name="t",
            vtype=gpy.GRB.CONTINUOUS,
            lb=-gpy.GRB.INFINITY,
            ub=gpy.GRB.INFINITY,
        )
        self.u = self.addVars(
            self.IJK,
            name="u",
            vtype=gpy.GRB.CONTINUOUS,
            lb=-gpy.GRB.INFINITY,
            ub=gpy.GRB.INFINITY,
        )

        # RESTRIÇÕES
        self.addConstrs(
            self.s[i, j, k] == self.r[i, j] * self.r[i, j] - self.dists[k] ** 2
            for i, j, k in self.IJK
        )
        self.addConstrs(
            self.t[i, j, k] == self.s[i, j, k] * self.s[i, j, k] for i, j, k in self.IJK
        )
        self.addConstrs(
            self.u[i, j, k] == self.a[i, j, k] * self.t[i, j, k] for i, j, k in self.IJK
        )
        # OBJETIVO
        if self.relaxed:
            self.setObjective(
                1
                + self.u.sum()
                + len(self.IJ)
                - gpy.quicksum(
                    self.a[i, j, k] * self.a[i, j, k] - 1 for i, j, k in self.IJK
                ),
                gpy.GRB.MINIMIZE,
            )
        else:
            self.setObjective(self.u.sum() + 1, gpy.GRB.MINIMIZE)

        self.update()
