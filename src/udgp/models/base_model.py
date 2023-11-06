"""
Gabriel Braun, 2023

Este módulo implementa o modelo base para instâncias do problema uDGP usando a biblioteca pyomo.
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
        time_limit=1e4,
        max_gap=1.0e-4,
        max_tol=1.0e-6,
        relaxed=False,
        previous_a: list | None = None,
    ):
        super(BaseModel, self).__init__()

        # PARÂMETROS DA INSTÂNCIA
        self.nx = pyo.Param(initialize=len(x_indices))
        self.ny = pyo.Param(initialize=len(y_indices))
        self.m = pyo.Param(initialize=len(dists))
        self.runtime = 0
        self.time_limit = time_limit

        # CONJUNTOS
        ## Conjunto I
        self.Iy = pyo.Set(initialize=y_indices)
        self.Ix = pyo.Set(initialize=x_indices)
        self.I = self.Iy | self.Ix

        ## Conjunto IJ
        self.IJyx = pyo.Set(initialize=self.Iy * self.Ix)
        self.IJxx = pyo.Set(
            within=self.Ix * self.Ix,
            initialize=((i, j) for i in self.Ix for j in self.Ix if i < j),
        )
        self.IJ = self.IJyx | self.IJxx

        ## Conjunto K
        all_k = np.arange(self.m)
        self.K = pyo.Set(initialize=[k for k in all_k if freqs[k] > 0])

        ## Conjunto L
        self.L = pyo.Set(initialize=[0, 1, 2])

        ## Conjunto A (soluções anteriores)
        n_previous_a = len(previous_a) if previous_a else 0
        self.A = pyo.Set(initialize=np.arange(n_previous_a))

        # PARÂMETROS
        self.max_gap = max_gap
        self.max_tol = max_tol
        self.max_err = self.max_gap + self.max_tol
        self.d_min = pyo.Param(initialize=dists[freqs != 0].min() - self.max_err)
        self.d_max = pyo.Param(initialize=dists[freqs != 0].max() + self.max_err)

        self.dists = pyo.Param(
            self.K,
            within=pyo.PositiveReals,
            initialize={k: dists[k] for k in self.K},
        )
        self.freqs = pyo.Param(
            self.K,
            within=pyo.NonNegativeIntegers,
            initialize={k: freqs[k] for k in self.K},
        )
        self.y = pyo.Param(
            self.Iy,
            self.L,
            within=pyo.Reals,
            initialize={(i, l): fixed_points[i, l] for i in self.Iy for l in self.L},
        )

        # VARIÁVEIS BASE
        self.relaxed = relaxed
        ## Decisão: distância k é referente ao par de átomos i e j
        if relaxed:
            self.a = pyo.Var(self.IJ, self.K, within=pyo.UnitInterval)
        else:
            self.a = pyo.Var(self.IJ, self.K, within=pyo.Binary)
        ## Coordenadas do ponto i
        self.x = pyo.Var(self.Ix, self.L, within=pyo.Reals)
        ## Vetor distância entre os átomos i e j
        self.v = pyo.Var(self.IJ, self.L, within=pyo.Reals)
        ## Distância entre os átomos i e j (norma de v)
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
        def constr_v_yx(self, i, j, l):
            return self.v[i, j, l] == self.y[i, l] - self.x[j, l]

        @self.Constraint(self.IJ)
        def constr_r(self, i, j):
            return self.r[i, j] ** 2 == sum(self.v[i, j, l] ** 2 for l in self.L)

        # RESTRIÇÕES PARA SOLUÇÕES ANTERIORES
        @self.Constraint(self.A)
        def constr_previous_a(self, n):
            return (
                sum(self.a[i, j, k] for i, j, k in previous_a[n])
                <= len(previous_a[n]) - 1
            )

    def solution_points(self):
        """
        Retorna (numpy.ndarray): pontos encontrados na solução do modelo.
        """
        return np.array([[self.x[i, l].value for l in self.L] for i in self.Ix])

    def solve(self, solver="gurobi", log=False):
        """
        Otimiza o modelo e atualiza a instância.

        Retorna (bool): verdadeiro se uma solução foi encontrada
        """
        opt = pyo.SolverFactory(solver, solver_io="python")

        # PARÂMETROS DO SOLVER
        mip_gap = self.max_gap * len(self.IJ)
        ## Gurobi
        if "gurobi" in solver.lower():
            opt.options["TimeLimit"] = self.time_limit
            opt.options["NonConvex"] = 2
            opt.options["MIPGapAbs"] = mip_gap
            opt.options["IntFeasTol"] = self.max_tol
            opt.options["FeasibilityTol"] = self.max_tol
            opt.options["OptimalityTol"] = self.max_tol
            opt.options["SolutionLimit "] = 1

        # OTIMIZA
        results = opt.solve(self, tee=log, report_timing=log)
        self.runtime = opt._solver_model.getAttr("Runtime")

        if results.solver.termination_condition == "infeasible":
            # self.status == "infeasible"
            return False

        return True
