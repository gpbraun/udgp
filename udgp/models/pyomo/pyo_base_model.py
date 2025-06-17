"""
Gabriel Braun, 2023

Este módulo implementa o modelo base para instâncias do problema uDGP usando a biblioteca pyomo.
"""

from itertools import combinations

import numpy as np
import pyomo.environ as pyo

from udgp.config import get_config


class pyoBaseModel(pyo.ConcreteModel):
    """
    Modelo base para o uDGP.
    """

    def __init__(
        self,
        *,
        x_indices: np.ndarray,
        y_indices: np.ndarray,
        dists: np.ndarray,
        freqs: np.ndarray,
        fixed_points: np.ndarray,
        previous_a: list | None = None,
        relaxed=False,
    ):
        super(pyoBaseModel, self).__init__()
        self.name = None

        # PARÂMETROS DA INSTÂNCIA
        self.nx = pyo.Param(initialize=len(x_indices))
        self.ny = pyo.Param(initialize=len(y_indices))
        self.m = pyo.Param(initialize=len(dists))
        self.runtime = 0

        # CONJUNTOS
        ## Conjunto I
        self.Iy = pyo.Set(initialize=y_indices)
        self.Ix = pyo.Set(initialize=x_indices)
        self.I = self.Iy | self.Ix

        ## Conjunto IJ
        self.IJyx = pyo.Set(initialize=self.Iy * self.Ix)
        self.IJxx = pyo.Set(initialize=combinations(self.Ix, 2))
        self.IJ = self.IJyx | self.IJxx

        ## Conjunto K
        all_k = np.arange(self.m)
        self.K = pyo.Set(initialize=all_k[freqs != 0])

        ## Conjunto L (dimensão)
        self.L = pyo.Set(initialize=[0, 1, 2])

        # PARÂMETROS
        self.d_min = pyo.Param(initialize=dists[freqs != 0].min())
        self.d_max = pyo.Param(initialize=dists[freqs != 0].max())

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
        def _constr_a1(self, k):
            return sum(self.a[i, j, k] for i, j in self.IJ) <= self.freqs[k]

        @self.Constraint(self.IJ)
        def _constr_a2(self, i, j):
            return sum(self.a[i, j, k] for k in self.K) == 1

        @self.Constraint(self.IJxx, self.L)
        def _constr_v_xx(self, i, j, l):
            return self.v[i, j, l] == self.x[j, l] - self.x[i, l]

        @self.Constraint(self.IJyx, self.L)
        def _constr_v_yx(self, i, j, l):
            return self.v[i, j, l] == self.y[i, l] - self.x[j, l]

        @self.Constraint(self.IJ)
        def _constr_r(self, i, j):
            return self.r[i, j] ** 2 == sum(self.v[i, j, l] ** 2 for l in self.L)

        # RESTRIÇÕES PARA SOLUÇÕES ANTERIORES
        n_previous_a = len(previous_a) if previous_a else 0
        self.A = pyo.Set(initialize=np.arange(n_previous_a))

        @self.Constraint(self.A)
        def _constr_previous_a(self, n):
            return (
                sum(self.a[i, j, k] for i, j, k in previous_a[n])
                <= len(previous_a[n]) - 1
            )

    def __setattr__(self, *args):
        try:
            return super(pyoBaseModel, self).__setattr__(*args)
        except AttributeError:
            return super(pyo.Model, self).__setattr__(*args)

    def solution_points(self) -> np.ndarray:
        """
        Retorna (numpy.ndarray): pontos encontrados na solução do modelo.
        """
        return np.array([[self.x[i, l].value for l in self.L] for i in self.Ix])

    def solve(
        self,
        solver="gurobi",
        *,
        log=False,
        config: dict | None = None,
        stage: str | None = None,
    ) -> bool:
        """
        Otimiza o modelo e atualiza a instância.

        Retorna (bool): verdadeiro se uma solução foi encontrada
        """
        opt = pyo.SolverFactory(solver, solver_io="python")

        config = get_config(
            solver=solver,
            model=self.name,
            stage=stage,
            overrides=config,
        )
        for k, v in config.items():
            opt.options[k] = v

        # OTIMIZA
        results = opt.solve(self, tee=log, report_timing=log)
        self.runtime = opt._solver_model.getAttr("Runtime")

        if results.solver.termination_condition == "infeasible":
            # self.status == "infeasible"
            return False

        return True
