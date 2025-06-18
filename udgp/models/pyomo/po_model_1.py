"""
Gabriel Braun, 2023

Este módulo implementa o modelo M1 para instâncias do problema uDGP.
"""

import pyomo.environ as po

from .po_base_model import poBaseModel


class poM1(poBaseModel):
    """Modelo M1 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(poM1, self).__init__(*args, **kwargs)
        self.name = "M1"

        # VARIÁVEIS
        self.s = po.Var(
            self.IJ,
            self.K,
            within=po.Reals,
        )
        self.t = po.Var(
            self.IJ,
            self.K,
            within=po.Reals,
        )
        self.u = po.Var(
            self.IJ,
            self.K,
            within=po.Reals,
        )

        # RESTRIÇÕES
        @self.Constraint(self.IJ, self.K)
        def _constr_x1(self, i, j, k):
            return self.s[i, j, k] == self.r[i, j] * self.r[i, j] - self.dists[k] ** 2

        @self.Constraint(self.IJ, self.K)
        def _constr_x2(self, i, j, k):
            return self.t[i, j, k] == self.s[i, j, k] * self.s[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def _constr_x3(self, i, j, k):
            return self.u[i, j, k] == self.a[i, j, k] * self.t[i, j, k]

        # OBJETIVO
        def relaxed_objective(model):
            return (
                1
                + len(model.IJ)
                + po.summation(model.u)
                - sum(model.a[i, j, k] ** 2 for i, j, k in model.IJ * model.K)
            )

        def objective(model):
            return 1 + po.summation(model.u)

        self.obj = po.Objective(
            sense=po.minimize,
            rule=relaxed_objective if self.relaxed else objective,
        )
