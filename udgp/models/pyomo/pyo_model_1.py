"""
Gabriel Braun, 2023

Este módulo implementa o modelo M1 para instâncias do problema uDGP.
"""

import pyomo.environ as pyo

from .pyo_base_model import pyoBaseModel


class pyoM1(pyoBaseModel):
    """Modelo M1 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(pyoM1, self).__init__(*args, **kwargs)
        self.name = "M1"

        # VARIÁVEIS
        self.s = pyo.Var(
            self.IJ,
            self.K,
            within=pyo.Reals,
        )
        self.t = pyo.Var(
            self.IJ,
            self.K,
            within=pyo.Reals,
        )
        self.u = pyo.Var(
            self.IJ,
            self.K,
            within=pyo.Reals,
        )

        # RESTRIÇÕES
        @self.Constraint(self.IJ, self.K)
        def constr_x1(self, i, j, k):
            return self.s[i, j, k] == self.r[i, j] * self.r[i, j] - self.dists[k] ** 2

        @self.Constraint(self.IJ, self.K)
        def constr_x2(self, i, j, k):
            return self.t[i, j, k] == self.s[i, j, k] * self.s[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def constr_x3(self, i, j, k):
            return self.u[i, j, k] == self.a[i, j, k] * self.t[i, j, k]

        # OBJETIVO
        def relaxed_objective(model):
            return (
                1
                + len(model.IJ)
                + pyo.summation(model.u)
                - sum(model.a[i, j, k] ** 2 for i, j, k in model.IJ * model.K)
            )

        def objective(model):
            return 1 + pyo.summation(model.u)

        self.obj = pyo.Objective(
            sense=pyo.minimize,
            rule=relaxed_objective if self.relaxed else objective,
        )
