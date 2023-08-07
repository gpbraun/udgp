"""Gabriel Braun, 2023
Este módulo implementa o modelo M1 para instâncias do problema uDGP.
"""

import gurobipy as gp
import numpy as np

from .gp_base_model import GPBaseModel


class M1gp(GPBaseModel):
    """Modelo M1 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(M1gp, self).__init__(*args, **kwargs)
        # VARIÁVEIS
        self.s = self.addVars(
            self.IJK,
            name="s",
            vtype=gp.GRB.CONTINUOUS,
            lb=-gp.GRB.INFINITY,
            ub=gp.GRB.INFINITY,
        )
        self.t = self.addVars(
            self.IJK,
            name="t",
            vtype=gp.GRB.CONTINUOUS,
            lb=-gp.GRB.INFINITY,
            ub=gp.GRB.INFINITY,
        )
        self.u = self.addVars(
            self.IJK,
            name="u",
            vtype=gp.GRB.CONTINUOUS,
            lb=-gp.GRB.INFINITY,
            ub=gp.GRB.INFINITY,
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
                - gp.quicksum(
                    self.a[i, j, k] * self.a[i, j, k] - 1 for i, j, k in self.IJK
                ),
                gp.GRB.MINIMIZE,
            )
        else:
            self.setObjective(self.u.sum() + 1, gp.GRB.MINIMIZE)

        self.update()
