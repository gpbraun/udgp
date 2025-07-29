"""
gp_model_2.py
"""

import gurobipy as gp

from .gp_base_model import gpBaseModel


class gpM2(gpBaseModel):
    """
    uDGP model M2.
    """

    NAME = "M2"
    PARAMS = {
        "MaxD2Gap": 5e-2,
    }

    def model_post_build(self):
        super().model_post_build()

        max_gap = self.ModelParams.MaxD2Gap

        # M2 VARIABLES
        ## p: calculated distance error
        self.p = self.addVars(
            self.IJ,
            name="p",
            vtype=gp.GRB.CONTINUOUS,
            lb=-max_gap,
            ub=max_gap,
        )
        ## w: calculated distance absolute error
        self.w = self.addVars(
            self.IJ,
            name="w",
            vtype=gp.GRB.CONTINUOUS,
            lb=0,
            ub=max_gap,
        )
        ## z: distância k se ela é referente ao par de átomos i e j. 0 em caso contrátrio.
        self.z = self.addVars(
            self.IJK,
            name="z",
            vtype=gp.GRB.SEMICONT,
            lb=self.d.min(),
            ub=self.d.max(),
            # vtype=gp.GRB.CONTINUOUS,
            # lb=0,
            # ub=self.d.max(),
        )

        # M2 CONSTRAINTS
        M = self.d.max()

        self._constr_m2_1 = self.addConstrs(
            self.w[i, j] >= self.p[i, j] for i, j in self.IJ
        )
        self._constr_m2_2 = self.addConstrs(
            self.w[i, j] >= -self.p[i, j] for i, j in self.IJ
        )
        self._constr_m2_3 = self.addConstrs(
            self.z[i, j, k] >= -(1 - self.a[i, j, k]) * M + self.r[i, j]
            for i, j, k in self.IJK
        )
        self._constr_m2_4 = self.addConstrs(
            self.z[i, j, k] <= (1 - self.a[i, j, k]) * M + self.r[i, j]
            for i, j, k in self.IJK
        )
        self._constr_m2_5 = self.addConstrs(
            self.z[i, j, k] >= -self.a[i, j, k] * M for i, j, k in self.IJK
        )
        self._constr_m2_6 = self.addConstrs(
            self.z[i, j, k] <= self.a[i, j, k] * M for i, j, k in self.IJK
        )
        self._constr_m2_7 = self.addConstrs(
            self.z[i, j, k] >= -(1 - self.a[i, j, k]) * M + self.d[k] + self.p[i, j]
            for i, j, k in self.IJK
        )
        self._constr_m2_8 = self.addConstrs(
            self.z[i, j, k] <= (1 - self.a[i, j, k]) * M + self.d[k] + self.p[i, j]
            for i, j, k in self.IJK
        )

        # M2 OBJECTIVE
        self.objective = 1 + self.w.sum()
