"""
Gabriel Braun, 2023

Este módulo implementa o modelo M2 para instâncias do problema uDGP.
"""

import pyomo.environ as po

from .po_base_model import poBaseModel


class poM2(poBaseModel):
    """Modelo M2 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(poM2, self).__init__(*args, **kwargs)
        self.name = "M2"

        # VARIÁVEIS
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
            # bounds=(0, self.d_max),
        )

        # RESTRIÇÕES
        @self.Constraint(self.IJ, self.K)
        def _constr_x1(self, i, j, k):
            return self.w[i, j, k] >= self.p[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def _constr_x2(self, i, j, k):
            return self.w[i, j, k] >= -self.p[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def _constr_x3(self, i, j, k):
            return self.z[i, j, k] >= self.r[i, j] - self.d_max * (1 - self.a[i, j, k])

        @self.Constraint(self.IJ, self.K)
        def _constr_x4(self, i, j, k):
            return self.z[i, j, k] <= self.r[i, j] + self.d_max * (1 - self.a[i, j, k])

        @self.Constraint(self.IJ, self.K)
        def _constr_x5(self, i, j, k):
            return self.z[i, j, k] >= -self.d_max * self.a[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def _constr_x6(self, i, j, k):
            return self.z[i, j, k] <= self.d_max * self.a[i, j, k]

        @self.Constraint(self.IJ, self.K)
        def _constr_x7(self, i, j, k):
            return self.z[i, j, k] >= self.dists[k] + self.p[i, j, k] - self.d_max * (
                1 - self.a[i, j, k]
            )

        @self.Constraint(self.IJ, self.K)
        def _constr_x8(self, i, j, k):
            return self.z[i, j, k] <= self.dists[k] + self.p[i, j, k] + self.d_max * (
                1 - self.a[i, j, k]
            )

        # OBJETIVO
        def relaxed_objective(model):
            return 1 + po.summation(model.w) - po.summation(model.a)

        def objective(model):
            return 1 + po.summation(model.w)

        self.obj = po.Objective(
            sense=po.minimize,
            rule=relaxed_objective if self.relaxed else objective,
        )
