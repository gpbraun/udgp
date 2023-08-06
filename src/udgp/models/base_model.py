"""
Gabriel Braun, 2023

Este módulo implementa o modelo base para instâncias do problema uDGP.
"""

import numpy as np
import pyomo.environ as pyo


class BaseModel(pyo.ConcreteModel):
    """
    Modelo base para o uDGP.
    """

    def __init__(
        self,
        x_indices: np.ndarray,
        y_indices: np.ndarray,
        dists: np.ndarray,
        freqs: np.ndarray,
        fixed_points: np.ndarray,
        max_gap=1e-2,
        max_tol=1e-4,
        relaxed=False,
    ):
        super(BaseModel, self).__init__()

        # PARÂMETROS DA INSTÂNCIA
        # self.instance = instance
        self.nx = pyo.Param(initialize=len(x_indices))
        self.ny = pyo.Param(initialize=len(y_indices))
        self.m = pyo.Param(initialize=len(dists))

        # CONJUNTOS
        ## Conjunto I
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
        k = np.arange(self.m)[freqs != 0]
        self.K = pyo.Set(initialize=k)

        ## Conjunto L
        self.L = pyo.Set(initialize=[0, 1, 2])

        # PARÂMETROS
        self.max_gap = pyo.Param(initialize=max_gap)
        self.max_tol = pyo.Param(initialize=max_tol)
        self.d_max = pyo.Param(initialize=dists[freqs != 0].max())
        self.d_min = pyo.Param(initialize=dists[freqs != 0].min())

        self.dists = pyo.Param(self.K, initialize=dists, within=pyo.PositiveReals)
        self.freqs = pyo.Param(self.K, initialize=freqs, within=pyo.NonNegativeIntegers)
        self.y = pyo.Param(
            self.Iy,
            self.L,
            within=pyo.Reals,
            initialize={(i, l): fixed_points[i, l] for i in self.Iy for l in self.L},
        )

        # VARIÁVEIS BASE
        self.relaxed = relaxed
        if relaxed:
            self.a = pyo.Var(self.IJ, self.K, within=pyo.UnitInterval)
        else:
            self.a = pyo.Var(self.IJ, self.K, within=pyo.Binary)

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

    def solution_points(self):
        """
        Retorna (np.ndarray): pontos encontrados na solução do modelo.
        """
        return np.array([[self.x[i, l].value for l in self.L] for i in self.Ix])

    def solve(self, solver="gurobi", log=False):
        """
        Otimiza o modelo e atualiza a instância.

        Retorna (bool): verdadeiro se uma solução foi encontrada
        """
        opt = pyo.SolverFactory(solver, solver_io="python")

        mip_gap = self.max_gap * len(self.IJ)

        if "gurobi" in solver.lower():
            opt.options["NonConvex"] = 2
            opt.options["MIPGapAbs"] = mip_gap
            opt.options["IntFeasTol"] = self.max_tol
            opt.options["FeasibilityTol"] = self.max_tol
            opt.options["OptimalityTol"] = self.max_tol
            opt.options["Cuts"] = 2

        opt.solve(self, tee=log, report_timing=log)
