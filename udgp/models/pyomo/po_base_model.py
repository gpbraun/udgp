"""
Gabriel Braun, 2023

Este módulo implementa o modelo base para instâncias do problema uDGP usando a biblioteca pyomo.
"""

import logging
from itertools import combinations

import numpy as np
import pyomo.environ as po

from udgp.solvers import get_config

logger = logging.getLogger(__name__)


class poBaseModel(po.ConcreteModel):
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
        super(poBaseModel, self).__init__()
        self.name = "Base"

        # PARÂMETROS DA INSTÂNCIA
        self.nx = po.Param(initialize=len(x_indices))
        self.ny = po.Param(initialize=len(y_indices))
        self.m = po.Param(initialize=len(dists))
        self.runtime = 0

        # CONJUNTOS
        ## Conjunto I
        self.Iy = po.Set(initialize=y_indices)
        self.Ix = po.Set(initialize=x_indices)
        self.I = self.Iy | self.Ix

        ## Conjunto IJ
        self.IJyx = self.Iy * self.Ix
        self.IJxx = po.Set(initialize=combinations(self.Ix, 2))
        self.IJ = self.IJyx | self.IJxx

        ## Conjunto K
        all_k = np.arange(self.m)
        self.K = po.Set(initialize=all_k[freqs != 0])

        ## Conjunto L (dimensão)
        self.L = po.Set(initialize=[0, 1, 2])

        # PARÂMETROS
        self.d_min = po.Param(initialize=dists[freqs != 0].min())
        self.d_max = po.Param(initialize=dists[freqs != 0].max())

        self.dists = po.Param(
            self.K,
            within=po.PositiveReals,
            initialize={k: dists[k] for k in self.K},
        )
        self.freqs = po.Param(
            self.K,
            within=po.NonNegativeIntegers,
            initialize={k: freqs[k] for k in self.K},
        )
        self.y = po.Param(
            self.Iy,
            self.L,
            within=po.Reals,
            initialize={(i, l): fixed_points[i, l] for i in self.Iy for l in self.L},
        )

        # VARIÁVEIS BASE
        self.relaxed = relaxed
        ## Decisão: distância k é referente ao par de átomos i e j
        if relaxed:
            self.a = po.Var(self.IJ, self.K, within=po.UnitInterval)
        else:
            self.a = po.Var(self.IJ, self.K, within=po.Binary)
        ## Coordenadas do ponto i
        self.x = po.Var(self.Ix, self.L, within=po.Reals)
        ## Vetor distância entre os átomos i e j
        self.v = po.Var(self.IJ, self.L, within=po.Reals)
        ## Distância entre os átomos i e j (norma de v)
        self.r = po.Var(self.IJ, within=po.Reals, bounds=(self.d_min, self.d_max))

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
        self.A = po.Set(initialize=np.arange(n_previous_a))

        @self.Constraint(self.A)
        def _constr_previous_a(self, n):
            return (
                sum(self.a[i, j, k] for i, j, k in previous_a[n])
                <= len(previous_a[n]) - 1
            )

    def __setattr__(self, *args):
        try:
            return super(poBaseModel, self).__setattr__(*args)
        except AttributeError:
            return super(po.Model, self).__setattr__(*args)

    def solution_points(self) -> np.ndarray:
        """
        Retorna (numpy.ndarray): pontos encontrados na solução do modelo.
        """
        return np.array([[self.x[i, l].value for l in self.L] for i in self.Ix])

    def solve(
        self,
        solver="gurobi",
        *,
        config: dict | None = None,
        stage: str | None = None,
    ) -> bool:
        """
        Otimiza o modelo e atualiza a instância.

        Retorna (bool): verdadeiro se uma solução foi encontrada
        """
        opt = po.SolverFactory(solver, solver_io="direct")
        config = get_config(
            solver=solver,
            model=self.name,
            stage=stage,
            overrides=config,
        )
        for k, v in config.items():
            opt.options[k] = v
        # OTIMIZA
        results = opt.solve(self, tee=logger)

        if solver == "gurobi":
            self.work = opt._solver_model.getAttr("Work")
            self.runtime = opt._solver_model.getAttr("Runtime")

        if results.solver.termination_condition == "infeasible":
            # self.status == "infeasible"
            return False

        return True
