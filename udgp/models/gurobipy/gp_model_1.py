"""Gabriel Braun, 2023
Este módulo implementa o modelo M1 para instâncias do problema uDGP.
"""

import gurobipy as gp

from .gp_base_model import gpBaseModel


class gpM1(gpBaseModel):
    """
    uDGP model M1.
    """

    NAME = "M1"

    def model_post_init(self):
        super().model_post_init()

        # M1 VARIABLES
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

        # M1 CONSTRAINTS
        self._constr_m1_1 = self.addConstrs(
            self.s[i, j, k] == self.r[i, j] * self.r[i, j] - self.dists[k] ** 2
            for i, j, k in self.IJK
        )
        self._constr_m1_2 = self.addConstrs(
            self.t[i, j, k] == self.s[i, j, k] * self.s[i, j, k] for i, j, k in self.IJK
        )
        self._constr_m1_3 = self.addConstrs(
            self.u[i, j, k] == self.a[i, j, k] * self.t[i, j, k] for i, j, k in self.IJK
        )

        # M1 OBJECTIVE
        self.objective = 1 + self.u.sum()
