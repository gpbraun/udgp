"""Gabriel Braun, 2023

Este módulo implementa o modelo M2 para instâncias do problema uDGP.
"""

import gurobipy as gp
import pyomo.environ as pyo
from gurobipy import GRB

from .base_gurobipy import BaseModelGurobipy
from .base_model import BaseModel


class M2(BaseModel):
    """Modelo M2 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(M2, self).__init__(*args, **kwargs)

        # VARIÁVEIS
        ## Erro no cálculo da distância
        self.p = pyo.Var(
            self.IJ,
            self.K,
            within=pyo.Reals,
            bounds=(-self.max_gap, self.max_gap),
        )
        ## Valor absoluto no erro do cálculo da distância
        self.w = pyo.Var(
            self.IJ,
            self.K,
            within=pyo.NonNegativeReals,
            bounds=(0, self.max_gap),
        )
        ## Distância k se ela é referente ao par de átomos i e j e 0 em caso contrátrio
        self.z = pyo.Var(
            self.IJ,
            self.K,
            within=pyo.NonNegativeReals,
            bounds=(0, self.d_max),
        )

        # RESTRIÇÕES
        @self.Constraint(self.IJ, self.K)
        def constr_x1(self, i, j, k):
            return self.w[i, j, k] >= self.p[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def constr_x2(self, i, j, k):
            return self.w[i, j, k] >= -self.p[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def constr_x3(self, i, j, k):
            return self.z[i, j, k] >= self.r[i, j] - self.d_max * (1 - self.a[i, j, k])

        @self.Constraint(self.IJ, self.K)
        def constr_x4(self, i, j, k):
            return self.z[i, j, k] <= self.r[i, j] + self.d_max * (1 - self.a[i, j, k])

        @self.Constraint(self.IJ, self.K)
        def constr_x5(self, i, j, k):
            return self.z[i, j, k] >= -self.d_max * self.a[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def constr_x6(self, i, j, k):
            return self.z[i, j, k] <= self.d_max * self.a[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def constr_x7(self, i, j, k):
            return self.z[i, j, k] >= self.dists[k] + self.p[i, j, k] - self.d_max * (
                1 - self.a[i, j, k]
            )

        @self.Constraint(self.IJ, self.K)
        def constr_x8(self, i, j, k):
            return self.z[i, j, k] <= self.dists[k] + self.p[i, j, k] + self.d_max * (
                1 - self.a[i, j, k]
            )

        # OBJETIVO
        def relaxed_objective(model):
            return 1 + pyo.summation(model.w) - pyo.summation(model.a)

        def objective(model):
            return 1 + pyo.summation(model.w)

        self.obj = pyo.Objective(
            sense=pyo.minimize,
            rule=relaxed_objective if self.relaxed else objective,
        )


class M2Gurobipy(BaseModelGurobipy):
    """Modelo M2 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(M2Gurobipy, self).__init__(*args, **kwargs)

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
