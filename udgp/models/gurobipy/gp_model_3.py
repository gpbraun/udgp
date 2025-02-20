"""Gabriel Braun, 2023
Este módulo implementa o modelo M1 para instâncias do problema uDGP.
"""

import gurobipy as gp
import numpy as np

from .gp_base_model import GPBaseModel


class M3gp(GPBaseModel):
    """
    Model 3 for the uDGP:
    A difference-of-convex formulation in Gurobi, based on
    r[i,j]^2 >= v[i,j] · v[i,j] and an objective that includes
    w[i,j,k]^2 + [r[i,j]^2 - v[i,j]·v[i,j]].
    """

    def __init__(self, *args, **kwargs):
        super(M3gp, self).__init__(*args, **kwargs)

        # ------------------------------------------------------
        # 1) Remove the original r^2 == v·v constraint
        #    so we can replace it with r^2 >= v·v.
        # ------------------------------------------------------
        # In gp_base_model.py, you added something like:
        #
        #  self.r_eq_constr = self.addConstrs(
        #      self.r[i, j]**2 == self.v[i, j] @ self.v[i, j] for i, j in self.IJ
        #  )
        #
        # If so, you can remove them by:
        #
        #    for c in self.r_eq_constr.values():
        #        self.remove(c)
        #
        # Or if they're unnamed, you need a custom approach:
        #    all_constrs = self.getConstrs()
        #    ... filter them out ...
        #
        # Below is a placeholder you can adapt to your naming scheme.

        # Example placeholder if you stored them in self._r_eq_constr:
        try:
            for c in self._r_eq_constr.values():
                self.remove(c)
        except AttributeError:
            pass  # If not stored, handle differently or skip.

        # ------------------------------------------------------
        # 2) Replace with the relaxed inequality:
        #    r[i,j]^2 >= (v[i,j]·v[i,j]).
        # ------------------------------------------------------
        # Gurobi's addQConstr allows direct quadratic constraints:
        for i, j in self.IJ:
            # r[i,j]^2 >= v[i,j]·v[i,j]
            self.addQConstr(
                self.r[i, j] * self.r[i, j] >= self.v[i, j] @ self.v[i, j],
                name=f"r_ge_v_{i}_{j}",
            )

        # ------------------------------------------------------
        # 3) Add new variables p, w, z (similar to model 2),
        #    but we will not do a linear objective of sum w,
        #    instead we do w^2 in the objective plus the
        #    difference-of-convex term [r^2 - v·v].
        # ------------------------------------------------------
        self.p = self.addVars(
            self.IJK,
            name="p",
            lb=-self.max_gap,
            ub=self.max_gap,
            vtype=gp.GRB.CONTINUOUS,
        )

        self.w = self.addVars(
            self.IJK, name="w", lb=0, ub=self.max_gap, vtype=gp.GRB.CONTINUOUS
        )

        self.z = self.addVars(
            self.IJK, name="z", lb=0, ub=self.d_max, vtype=gp.GRB.CONTINUOUS
        )

        # ------------------------------------------------------
        # 4) Big-M constraints that tie (p, w, z) to a[i,j,k]
        #    and r[i,j], similar to model 2, but unchanged.
        # ------------------------------------------------------
        # |p[i,j,k]| <= w[i,j,k]
        self.addConstrs((self.w[i, j, k] >= self.p[i, j, k]) for (i, j, k) in self.IJK)
        self.addConstrs((self.w[i, j, k] >= -self.p[i, j, k]) for (i, j, k) in self.IJK)

        # z[i,j,k] = 0 if a[i,j,k] = 0; else approx r[i,j] ~ dists[k]+p[i,j,k]
        self.addConstrs(
            (self.z[i, j, k] >= self.r[i, j] - (1 - self.a[i, j, k]) * self.d_max)
            for (i, j, k) in self.IJK
        )
        self.addConstrs(
            (self.z[i, j, k] <= self.r[i, j] + (1 - self.a[i, j, k]) * self.d_max)
            for (i, j, k) in self.IJK
        )
        self.addConstrs(
            (self.z[i, j, k] >= -self.d_max * self.a[i, j, k]) for (i, j, k) in self.IJK
        )
        self.addConstrs(
            (self.z[i, j, k] <= self.d_max * self.a[i, j, k]) for (i, j, k) in self.IJK
        )
        self.addConstrs(
            (
                self.z[i, j, k]
                >= self.dists[k] + self.p[i, j, k] - (1 - self.a[i, j, k]) * self.d_max
            )
            for (i, j, k) in self.IJK
        )
        self.addConstrs(
            (
                self.z[i, j, k]
                <= self.dists[k] + self.p[i, j, k] + (1 - self.a[i, j, k]) * self.d_max
            )
            for (i, j, k) in self.IJK
        )

        # ------------------------------------------------------
        # 5) Define the difference-of-convex objective:
        #
        #    1
        #  + ∑( w[i,j,k]^2 )
        #  + ∑( r[i,j]^2 - v[i,j]·v[i,j] ).
        #
        # Gurobi’s objective can handle quadratic terms.
        # We'll build it via a LinExpr or QuadExpr.
        # ------------------------------------------------------
        obj = gp.QuadExpr()

        # Constant 1
        obj += 1.0

        # sum of w[i,j,k]^2
        for i, j, k in self.IJK:
            obj += self.w[i, j, k] * self.w[i, j, k]

        # sum over i,j of r[i,j]^2 minus sum(v[i,j]^2)
        for i, j in self.IJ:
            # r[i,j]^2
            obj += self.r[i, j] * self.r[i, j]
            # subtract (v[i,j,0]^2 + v[i,j,1]^2 + v[i,j,2]^2)
            # self.v[i, j] is a 3D MVar, so do each component:
            for comp in range(3):
                obj -= self.v[i, j][comp] * self.v[i, j][comp]

        # Set the new objective (non-convex quadratic)
        self.setObjective(obj, gp.GRB.MINIMIZE)

        self.update()
