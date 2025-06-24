"""
Gabriel Braun, 2023

Este módulo implementa o modelo M1 para instâncias do problema uDGP.
"""

import pyomo.environ as po

from .po_base_model import poBaseModel


class poM1(poBaseModel):
    """
    uDGP model M1.
    """

    NAME = "M1"

    def model_post_init(self):
        super().model_post_init()

        # M1 VARIABLES
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

        # M1 CONSTRAINTS
        @self.Constraint(self.IJ, self.K)
        def _constr_m1_1(self, i, j, k):
            return self.s[i, j, k] == self.r[i, j] ** 2 - self.dists[k] ** 2

        @self.Constraint(self.IJ, self.K)
        def _constr_m1_2(self, i, j, k):
            return self.t[i, j, k] == self.s[i, j, k] ** 2

        @self.Constraint(self.IJ, self.K)
        def _constr_m1_3(self, i, j, k):
            return self.u[i, j, k] == self.a[i, j, k] * self.t[i, j, k]

        # M1 OBJECTIVE
        self.objective = 1 + po.summation(self.u)
