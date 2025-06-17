"""Gabriel Braun, 2023
Este módulo implementa o modelo M1 para instâncias do problema uDGP.
"""

import gurobipy as gpy

from .gpy_base_model import gpyBaseModel


class gpyM2(gpyBaseModel):
    """Modelo M2 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(gpyM2, self).__init__(*args, **kwargs)
        self.name = "M2"

        # VARIÁVEIS
        ## Erro no cálculo da distância
        self.p = self.addVars(
            self.IJK,
            name="p",
            vtype=gpy.GRB.CONTINUOUS,
            lb=-gpy.GRB.INFINITY,
            ub=gpy.GRB.INFINITY,
        )
        ## Valor absoluto no erro do cálculo da distância
        self.w = self.addVars(
            self.IJK,
            name="w",
            vtype=gpy.GRB.CONTINUOUS,
            lb=0,
            ub=gpy.GRB.INFINITY,
        )
        ## Distância k se ela é referente ao par de átomos i e j. 0 em caso contrátrio.
        self.z = self.addVars(
            self.IJK,
            name="z",
            vtype=gpy.GRB.CONTINUOUS,
            ub=self.d_max,
        )

        # RESTRIÇÕES
        self._constr_x1 = self.addConstrs(
            self.w[i, j, k] >= self.p[i, j, k] for i, j, k in self.IJK
        )
        self._constr_x2 = self.addConstrs(
            self.w[i, j, k] >= -self.p[i, j, k] for i, j, k in self.IJK
        )
        self._constr_x3 = self.addConstrs(
            self.z[i, j, k] >= -(1 - self.a[i, j, k]) * self.d_max + self.r[i, j]
            for i, j, k in self.IJK
        )
        self._constr_x4 = self.addConstrs(
            self.z[i, j, k] <= (1 - self.a[i, j, k]) * self.d_max + self.r[i, j]
            for i, j, k in self.IJK
        )
        self._constr_x5 = self.addConstrs(
            self.z[i, j, k] >= -self.a[i, j, k] * self.d_max for i, j, k in self.IJK
        )
        self._constr_x6 = self.addConstrs(
            self.z[i, j, k] <= self.a[i, j, k] * self.d_max for i, j, k in self.IJK
        )
        self._constr_x7 = self.addConstrs(
            self.z[i, j, k]
            >= -(1 - self.a[i, j, k]) * self.d_max + self.dists[k] + self.p[i, j, k]
            for i, j, k in self.IJK
        )
        self._constr_x8 = self.addConstrs(
            self.z[i, j, k]
            <= (1 - self.a[i, j, k]) * self.d_max + self.dists[k] + self.p[i, j, k]
            for i, j, k in self.IJK
        )

        # OBJETIVO
        if self.relaxed:
            self.setObjective(
                1
                + self.w.sum()
                - gpy.quicksum(
                    self.a[i, j, k] * self.a[i, j, k] for i, j, k in self.IJK
                ),
                gpy.GRB.MINIMIZE,
            )
        else:
            self.setObjective(1 + self.w.sum(), gpy.GRB.MINIMIZE)
