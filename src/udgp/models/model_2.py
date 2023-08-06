"""
Gabriel Braun, 2023

Este módulo implementa o modelo M2 para instâncias do problema uDGP.
"""

import pyomo.environ as pyo

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
            return self.z[i, j, k] >= self.dists[k] * (
                1 + self.p[i, j, k]
            ) - self.d_max * (1 - self.a[i, j, k])

        @self.Constraint(self.IJ, self.K)
        def constr_x8(self, i, j, k):
            return self.z[i, j, k] <= self.dists[k] * (
                1 + self.p[i, j, k]
            ) + self.d_max * (1 - self.a[i, j, k])

        # OBJETIVO
        def relaxed_objective(model):
            return 1 + pyo.summation(model.w) - pyo.summation(model.a)

        def objective(model):
            return 1 + pyo.summation(model.w)

        self.obj = pyo.Objective(
            sense=pyo.minimize,
            rule=relaxed_objective if self.relaxed else objective,
        )
