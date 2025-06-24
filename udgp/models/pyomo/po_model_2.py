"""
Gabriel Braun, 2023

Este módulo implementa o modelo M2 para instâncias do problema uDGP.
"""

import pyomo.environ as po

from .po_base_model import poBaseModel


class poM2(poBaseModel):
    """
    uDGP model M2.
    """

    NAME = "M2"

    def model_post_init(self):
        super().model_post_init()

        # M2 VARIABLES
        ## Erro no cálculo da distância
        self.p = po.Var(
            self.IJ,
            self.K,
            within=po.Reals,
        )
        ## Valor absoluto no erro do cálculo da distância
        self.w = po.Var(
            self.IJ,
            self.K,
            within=po.NonNegativeReals,
        )
        ## Distância k se ela é referente ao par de átomos i e j e 0 em caso contrátrio
        self.z = po.Var(
            self.IJ,
            self.K,
            within=po.NonNegativeReals,
            bounds=(0, self.d_max),
        )

        # M2 CONSTRAINTS
        @self.Constraint(self.IJ, self.K)
        def _constr_m2_1(self, i, j, k):
            return self.w[i, j, k] >= self.p[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def _constr_m2_2(self, i, j, k):
            return self.w[i, j, k] >= -self.p[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def _constr_m2_3(self, i, j, k):
            return self.z[i, j, k] >= self.r[i, j] - self.d_max * (1 - self.a[i, j, k])

        @self.Constraint(self.IJ, self.K)
        def _constr_m2_4(self, i, j, k):
            return self.z[i, j, k] <= self.r[i, j] + self.d_max * (1 - self.a[i, j, k])

        @self.Constraint(self.IJ, self.K)
        def _constr_m2_5(self, i, j, k):
            return self.z[i, j, k] >= -self.d_max * self.a[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def _constr_m2_6(self, i, j, k):
            return self.z[i, j, k] <= self.d_max * self.a[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def _constr_m2_7(self, i, j, k):
            return self.z[i, j, k] >= self.dists[k] + self.p[i, j, k] - self.d_max * (
                1 - self.a[i, j, k]
            )

        @self.Constraint(self.IJ, self.K)
        def _constr_m2_8(self, i, j, k):
            return self.z[i, j, k] <= self.dists[k] + self.p[i, j, k] + self.d_max * (
                1 - self.a[i, j, k]
            )

        # M2 OBJECTIVE
        self.objective = 1 + po.summation(self.w)
