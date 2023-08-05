"""Gabriel Braun, 2023

Este módulo implementa o modelo M1 para instâncias do problema uDGP.
"""

import gurobipy as gp
import pyomo.environ as pyo

from .base_gurobipy import BaseModelGurobipy
from .base_model import BaseModel


class M1(BaseModelGurobipy):
    """Modelo M1 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(M1, self).__init__(*args, **kwargs)

        # VARIÁVEIS
        self.s = self.addVars(
            self.ijk_indices(),
            name="s",
            vtype=gp.GRB.CONTINUOUS,
            lb=-gp.GRB.INFINITY,
            ub=gp.GRB.INFINITY,
        )
        self.t = self.addVars(
            self.ijk_indices(),
            name="t",
            vtype=gp.GRB.CONTINUOUS,
            lb=-gp.GRB.INFINITY,
            ub=gp.GRB.INFINITY,
        )
        self.u = self.addVars(
            self.ijk_indices(),
            name="u",
            vtype=gp.GRB.CONTINUOUS,
            lb=-gp.GRB.INFINITY,
            ub=gp.GRB.INFINITY,
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
                + len(list(self.ij_indices()))
                - gp.quicksum(
                    self.a[i, j, k] * self.a[i, j, k] - 1
                    for i, j, k in self.ijk_indices()
                ),
                gp.GRB.MINIMIZE,
            )
        else:
            self.setObjective(self.u.sum() + 1, gp.GRB.MINIMIZE)

        self.update()
