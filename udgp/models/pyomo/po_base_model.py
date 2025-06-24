"""
Gabriel Braun, 2023

Este módulo implementa o modelo base para instâncias do problema uDGP usando a biblioteca pyomo.
"""

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pyomo.environ as po

from udgp.solvers import get_solver_params

logger = logging.getLogger("uDGP")


class poBaseModel(po.ConcreteModel):
    """
    uDGP base model.
    """

    NAME = "Base"
    PARAMS = {
        "Lambda": 1,
    }

    def _add_core_vars(self):
        """
        Add the variables to the models.
        """
        ## Decisão: distância k é referente ao par de átomos i e j
        self.a = po.Var(
            self.IJ,
            self.K,
            within=po.Binary,
        )
        ## Coordenadas do ponto i
        self.x = po.Var(
            self.Ix,
            self.L,
            within=po.Reals,
        )
        ## Vetor distância entre os átomos i e j
        self.v = po.Var(
            self.IJ,
            self.L,
            within=po.Reals,
        )
        ## Distância entre os átomos i e j (norma de v)
        self.r = po.Var(
            self.IJ,
            within=po.Reals,
            bounds=(self.d_min, self.d_max),
        )

    def _add_core_constrs(self):
        """
        Adds the constraints to the model
        """

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

        ## Constraint to take previous solutions into account
        n_previous_a = len(self.previous_a) if self.previous_a else 0
        self.A = po.Set(initialize=np.arange(n_previous_a))

        @self.Constraint(self.A)
        def _constr_previous_a(self, n):
            a_ijk_idx = self.previous_a[n]
            return sum(self.a[ijk] for ijk in a_ijk_idx) <= len(a_ijk_idx) - 1

    @property
    def objective(self):
        """
        Returns: objective expression.
        """
        return self._objective.expr

    @objective.setter
    def objective(self, expr):
        """
        Sets: objective expression.
        """
        self._objective = po.Objective(sense=po.minimize, expr=expr)

    def set_model_params(
        self,
        overrides: dict[str, Any] | None = None,
    ):
        """
        Sets: model-specific params.
        """
        self.model_params = dict(self.PARAMS)
        if overrides is not None:
            self.model_params.update(overrides)

    def model_post_init(self):
        """
        Run *after* the base constructor finishes.

        Subclasses add their own logic here and should call super().
        """
        return

    def __init__(
        self,
        *,
        x_indices: np.ndarray,
        y_indices: np.ndarray,
        dists: np.ndarray,
        freqs: np.ndarray,
        fixed_points: np.ndarray,
        model_params: dict | None = None,
        previous_a: list | None = None,
    ):
        super(poBaseModel, self).__init__()

        self.set_model_params(overrides=model_params)

        self.total_runtime = 0
        self.total_work = 0

        # INSTANCE ATTRS
        self.m = po.Param(initialize=len(dists))
        self.nx = po.Param(initialize=len(x_indices))
        self.ny = po.Param(initialize=len(y_indices))

        # INSTANCE SETS
        ## Dimension Set
        self.L = po.Set(initialize=[0, 1, 2])
        ## I Sets
        self.Iy = po.Set(initialize=y_indices.tolist())
        self.Ix = po.Set(initialize=x_indices.tolist())
        self.I = self.Iy | self.Ix
        ## IJ Sets
        self.IJyx = self.Iy * self.Ix
        self.IJxx = po.Set(initialize=combinations(self.Ix, 2))
        self.IJ = self.IJyx | self.IJxx
        ## K Set
        self.K = po.Set(initialize=np.arange(self.m)[freqs != 0].tolist())
        self.IJK = self.IJ * self.K

        # OTHER INSTANCE ATTRS
        self.y = po.Param(
            self.Iy,
            self.L,
            within=po.Reals,
            initialize={(i, l): fixed_points[i, l] for i in self.Iy for l in self.L},
        )
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
        self.d_min = po.Param(initialize=dists[freqs != 0].min())
        self.d_max = po.Param(initialize=dists[freqs != 0].max())

        self.previous_a = previous_a if previous_a is not None else []

        # VARIABLES AND CONSTRAINTS
        self._add_core_vars()
        self._add_core_constrs()
        self.model_post_init()

    def solution_points(self) -> np.ndarray:
        """
        Retorna (numpy.ndarray): pontos encontrados na solução do modelo.
        """
        return np.array([[self.x[i, l].value for l in self.L] for i in self.Ix])

    def relax_a(self):
        """
        Relax variables.
        """
        self.a.domain = po.UnitInterval

        lbd = self.model_params["Lambda"]

        self.objective += lbd * (
            len(self.IJ) - sum(self.a[ijk] ** 2 for ijk in self.IJK)
        )

    def _get_solver(
        self,
        solver: str,
        *,
        stage: str | None = None,
        overrides: dict[str, Any] | None = None,
    ):
        """
        Returns: solver with specific params.
        """
        opt = po.SolverFactory(solver, solver_io="direct")

        params = get_solver_params(
            solver=solver,
            model=self.NAME,
            stage=stage,
        )
        if overrides is not None:
            params.update(overrides)

        for k, v in params.items():
            opt.options[k] = v

        return opt

    def solve(
        self,
        solver="gurobi",
        *,
        stage: str | None = None,
        solver_params: dict | None = None,
    ) -> bool:
        """
        Otimiza o modelo e atualiza a instância.

        Retorna (bool): verdadeiro se uma solução foi encontrada
        """
        opt = self._get_solver(solver, stage=stage, overrides=solver_params)
        results = opt.solve(self, tee=logger)

        if solver == "gurobi":
            self.total_runtime += opt._solver_model.getAttr("Runtime")
            self.total_work += opt._solver_model.getAttr("Work")

        if results.solver.termination_condition == "infeasible":
            return False

        return True
