"""Gabriel Braun, 2023

Este módulo implementa o selfo base para instâncias do problema uDGP usando a API pyomo.
"""

import numpy as np
import pyomo.environ as pyo

from udgp.instances.instance import Instance


class BaseModel(pyo.ConcreteModel):
    """
    Modelo base para o uDGP usando a API pyomo.
    """

    def __init__(
        self,
        instance: Instance,
        nx: int | None = None,
        ny: int | None = None,
        max_gap=5e-3,
        max_tol=1e-4,
        relaxed=False,
    ):
        super(BaseModel, self).__init__()

        # PARÂMETROS DA INSTÂNCIA
        self.instance = instance
        self.n = pyo.Param(initialize=instance.n)
        self.m = pyo.Param(initialize=instance.m)

        # CONJUNTOS
        ## Conjunto I
        rng = np.random.default_rng()
        y_indices = rng.choice(instance.fixed_n, ny, replace=False)
        x_indices = np.arange(instance.fixed_n, nx + instance.fixed_n)

        self.Iy = pyo.Set(initialize=y_indices)
        self.Ix = pyo.Set(initialize=x_indices)
        self.I = self.Iy | self.Ix

        ## Conjunto J
        self.IJxx = pyo.Set(
            within=self.Ix * self.Ix,
            initialize=((i, j) for i in self.Ix for j in self.Ix if i < j),
        )
        self.IJyx = pyo.Set(initialize=self.Iy * self.Ix)

        self.IJ = self.IJyx | self.IJxx

        ## Conjunto K
        k = np.arange(instance.m)[instance.freqs != 0]
        self.K = pyo.Set(initialize=k)

        ## Conjunto L
        self.L = pyo.Set(initialize=[0, 1, 2])

        # PARÂMETROS
        self.max_gap = pyo.Param(initialize=max_gap)
        self.max_tol = pyo.Param(initialize=max_tol)
        self.d_max = pyo.Param(initialize=instance.dists.max())
        self.d_min = pyo.Param(initialize=instance.dists.min())

        self.dists = pyo.Param(
            self.K,
            within=pyo.PositiveReals,
            initialize=instance.dists,
        )
        self.freqs = pyo.Param(
            self.K,
            within=pyo.NonNegativeIntegers,
            initialize=instance.freqs,
        )
        self.y = pyo.Param(
            self.Iy,
            self.L,
            within=pyo.Reals,
            initialize={(i, l): instance.points[i, l] for i in self.Iy for l in self.L},
        )

        # VARIÁVEIS BASE
        if relaxed:
            self.a = pyo.Var(
                self.IJ,
                self.K,
                within=pyo.UnitInterval,
            )
        else:
            self.a = pyo.Var(
                self.IJ,
                self.K,
                within=pyo.Binary,
            )
        self.x = pyo.Var(self.Ix, self.L, within=pyo.Reals)
        self.v = pyo.Var(self.IJ, self.L, within=pyo.Reals)
        self.r = pyo.Var(self.IJ, within=pyo.Reals, bounds=(self.d_min, self.d_max))

        # RESTRIÇÕES BASE
        @self.Constraint(self.K)
        def constr_a1(self, k):
            return sum(self.a[i, j, k] for i, j in self.IJ) <= self.freqs[k]

        @self.Constraint(self.IJ)
        def constr_a2(self, i, j):
            return sum(self.a[i, j, k] for k in self.K) == 1

        @self.Constraint(self.IJxx, self.L)
        def constr_v_xx(self, i, j, l):
            return self.v[i, j, l] == self.x[j, l] - self.x[i, l]

        @self.Constraint(self.IJyx, self.L)
        def constr_v_xy(self, i, j, l):
            return self.v[i, j, l] == self.y[i, l] - self.x[j, l]

        @self.Constraint(self.IJ)
        def constr_r(self, i, j):
            return self.r[i, j] ** 2 == sum(self.v[i, j, l] ** 2 for l in self.L)

    def optimize(self, solver="gurobi_direct", log=False):
        """
        Otimiza o modelo e atualiza a instância.

        Retorna: verdadeiro se uma solução foi encontrada
        """
        opt = pyo.SolverFactory(solver)

        if solver in ["gurobi", "gurobi_direct"]:
            opt.options["NonConvex"] = 2
            opt.options["MIPGap"] = self.max_gap * len(self.IJ.data())
            opt.options["IntFeasTol"] = self.max_tol
            opt.options["FeasibilityTol"] = self.max_tol
            opt.options["OptimalityTol"] = self.max_tol

        opt.solve(self, tee=log, report_timing=log)

        # ATUALIZA A INSTÂNCIA
        new_points = np.array([[self.x[i, l].value for l in self.L] for i in self.Ix])
        if not self.instance.add_points(new_points, 2 * self.max_gap):
            return False
        else:
            return True
