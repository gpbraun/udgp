"""
Gabriel Braun, 2024

Este módulo implementa o modelo M3 para instâncias do problema uDGP.
"""

import pyomo.environ as pyo

from .base_model import BaseModel


class M3(BaseModel):
    """Modelo M2 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(M3, self).__init__(*args, **kwargs)

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
        self.del_component(self.constr_r)

        @self.Constraint(self.IJ)
        def constr_r(self, i, j):
            return self.r[i, j] ** 2 >= sum(self.v[i, j, l] ** 2 for l in self.L)

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
        def objective(model):
            return (
                1
                + sum(self.w[i, j, k] ** 2 for i, j, k in model.IJ * model.K)
                + sum(
                    self.r[i, j] ** 2 - sum(self.v[i, j, l] ** 2 for l in self.L)
                    for i, j in self.IJ
                )
            )

        self.obj = pyo.Objective(
            sense=pyo.minimize,
            rule=objective,
        )
