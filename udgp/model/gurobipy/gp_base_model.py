"""
gp_base_model.py
"""

import logging
from itertools import chain, combinations, product
from typing import Any

import gurobipy as gp
import numpy as np

from udgp.solver import get_solver_params
from udgp.utils.params import ParamView

logger = logging.getLogger("udgp")


class gpBaseModel:
    """
    uDGP base model.
    """

    NAME = "Base"
    PARAMS = {
        "Relax": 0,
        "Lambda": 1,
    }

    # ==================================================================================
    #   HELPERS
    # ==================================================================================

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        all_params = {}
        for model in reversed(cls.mro()):
            all_params.update(getattr(model, "PARAMS", {}))

        cls._PARAMS = all_params

    # ==================================================================================
    #   CONSTRUCTOR
    # ==================================================================================

    def __init__(
        self,
        *,
        x_indices: np.ndarray,
        y_indices: np.ndarray,
        dists: np.ndarray,
        freqs: np.ndarray,
        fixed_points: np.ndarray,
        model_params: dict | None = None,
        env: gp.Env | None = None,
    ):
        self.time = 0.0
        self.work = 0.0

        # SETS AND INSTANCE DATA
        self.m = len(dists)
        self.nx = len(x_indices)
        self.ny = len(y_indices)

        ## index sets
        self.Iy = gp.tuplelist(y_indices.tolist())
        self.Ix = gp.tuplelist(x_indices.tolist())
        self.I = gp.tuplelist(chain(self.Iy, self.Ix))

        self.IJyx = gp.tuplelist(product(self.Iy, self.Ix))
        self.IJxx = gp.tuplelist(combinations(self.Ix, 2))
        self.IJ = gp.tuplelist(chain(self.IJyx, self.IJxx))

        self.K = gp.tuplelist([k for k in range(self.m) if freqs[k] > 0])
        self.IJK = gp.tuplelist([(i, j, k) for (i, j), k in product(self.IJ, self.K)])

        ## data
        self.y = gp.tupledict((i, fixed_points[i]) for i in y_indices)
        self.d = dists**2
        self.f = freqs

        # BUILD INTERNAL GUROBIPY MODEL
        self._model = None

        self._env = env or gp.Env(empty=True)
        self._env.setParam("LogToConsole", 0)
        self._env.start()

        self._model_build(model_params)

    # ==================================================================================
    #   META PARAMETER HELPERS
    # ==================================================================================

    def _set_model_params(
        self,
        overrides: dict[str, Any] | None = None,
    ):
        """
        Sets: model-specific params.
        """
        self._model_params = dict(self._PARAMS)
        if overrides is not None:
            self._model_params.update(overrides)

        self.ModelParams = ParamView(self._model_params)

    # ==================================================================================
    #   BUILD MODEL
    # ==================================================================================

    def _model_build(
        self,
        model_params: dict | None = None,
    ):
        """
        Build internal `gurobipy` model.
        """
        if self._model:
            self._model.dispose()

        self._model = gp.Model(f"uDGP_{self.NAME}", self._env)

        self._set_model_params(overrides=model_params)

        ## variables and constraints
        self._model_add_core_vars()
        self._model_add_core_constrs()
        self.model_post_build()

        self.update()

    # ==================================================================================
    #   VARIABLES AND CONSTRAINTS
    # ==================================================================================

    def _model_add_core_vars(self):
        """
        Adds: core variables to the models.
        """
        # Decisão: distância k é referente ao par de átomos i e j
        self.a = self.addVars(
            self.IJK,
            name="a",
            vtype=gp.GRB.CONTINUOUS if self.ModelParams.Relax else gp.GRB.BINARY,
            lb=0,
            ub=1,
        )
        self.a.BranchPriority = 10
        # Coordenadas do ponto i
        self.x = gp.tupledict(
            (
                i,
                self.addMVar(
                    3,
                    name=f"x[{i}]",
                    vtype=gp.GRB.CONTINUOUS,
                    lb=-gp.GRB.INFINITY,
                    ub=gp.GRB.INFINITY,
                ),
            )
            for i in self.Ix
        )
        # Vetor distância entre os átomos i e j
        self.v = gp.tupledict(
            (
                ij,
                self.addMVar(
                    3,
                    name=f"v[{ij}]",
                    vtype=gp.GRB.CONTINUOUS,
                    lb=-gp.GRB.INFINITY,
                    ub=gp.GRB.INFINITY,
                ),
            )
            for ij in self.IJ
        )
        # Distância entre os átomos i e j (norma de v)
        self.r = self.addVars(
            self.IJ,
            name="r",
            vtype=gp.GRB.CONTINUOUS,
            lb=self.d.min(),
            ub=self.d.max(),
        )

    def _model_add_core_constrs(self):
        """
        Adds: core constraints to the model
        """
        self._constr_a1 = self.addConstrs(
            self.a.sum("*", "*", k) <= self.f[k] for k in self.K
        )
        self._constr_a2 = self.addConstrs(
            self.a.sum(i, j, "*") == 1 for i, j in self.IJ
        )
        self._constr_v_xx = self.addConstrs(
            self.v[i, j] == self.x[i] - self.x[j] for i, j in self.IJxx
        )
        self._constr_v_yx = self.addConstrs(
            self.v[i, j] == self.y[i] - self.x[j] for i, j in self.IJyx
        )
        self._constr_r = self.addConstrs(
            self.v[i, j] @ self.v[i, j] == self.r[i, j] for i, j in self.IJ
        )

    # ==================================================================================
    #   OBJECTIVE
    # ==================================================================================

    @property
    def objective(self):
        """
        Returns: objective expression.
        """
        return self._objective

    @objective.setter
    def objective(self, expr):
        """
        Sets: objective expression.
        """
        self._objective = expr
        self.setObjective(expr, gp.GRB.MINIMIZE)

    # ==================================================================================
    #   LIFE-CYCLE HOOKS FOR SUBCLASSES
    # ==================================================================================

    def model_post_build(self):
        """
        Run *after* the base constructor finishes.

        Subclasses add their own logic here and should call super().
        """
        return

    def model_pre_solve(self):
        """
        Run *before* the solve() routine.

        Subclasses add their own logic here and should call super().
        """
        if self.ModelParams.Relax:
            lbd = self.ModelParams.Lambda
            self.objective += lbd * (
                len(self.IJ) - gp.quicksum(a**2 for a in self.a.values())
            )

    # ==================================================================================
    #   SOLVER PARAMS
    # ==================================================================================

    def _set_solver_params(
        self,
        *,
        stage: str | None = None,
        overrides: dict[str, Any] | None = None,
    ):
        """
        Sets: solver-specific params.
        """
        self.resetParams()

        params = get_solver_params(
            solver="gurobi",
            model=self.NAME,
            stage=stage,
        )
        if overrides is not None:
            params.update(overrides)

        self.Params.OutputFlag = 0

        for k, v in params.items():
            try:
                self.setParam(k, v)
            except Exception:
                logger.warning(f"Error setting Gurobi param: {k} = {v}.")

        self.Params.OutputFlag = 1

    # ==================================================================================
    #   MODEL SOLVERS
    # ==================================================================================

    def optimize(self):
        """
        Optimize current model.
        """
        self.model_pre_solve()
        self._model.optimize()

        self.time += self._model.Runtime
        self.work += self._model.Work

    def solve(
        self,
        *,
        stage: str | None = None,
        solver_params: dict | None = None,
    ):
        """
        Solve uDGP instance.
        """
        self._set_solver_params(stage=stage, overrides=solver_params)

        self.optimize()

        if self.SolCount == 0:
            return False

        return True

    # ==================================================================================
    #   SOLUTION HELPERS
    # ==================================================================================

    @property
    def assignments(self) -> dict[tuple[int], int]:
        """
        Retorna (dict[tuple[int], int]): indices que correspondem aos valores de a unitários.
        """
        return {
            (i, j): k
            for (i, j, k), a in self.a.items()
            if np.isclose(a.X, 1, atol=1e-2)
        }

    @property
    def sol_x(self) -> dict:
        """
        Retorna (dict): mapa de valores da variável x.
        """
        return {idx: x.X for idx, x in self.x.items()}

    @property
    def sol_v(self) -> dict:
        """
        Retorna (dict): mapa de valores da variável v.
        """
        return {idx: v.X for idx, v in self.v.items()}

    @property
    def sol_x_array(self) -> np.ndarray:
        """
        Retorna (numpy.ndarray): valores da variável x.
        """
        return np.array([x.X for x in self.x.values()])

    @property
    def sol_v_array(self) -> np.ndarray:
        """
        Retorna (numpy.ndarray): valores da variável v.
        """
        return np.array([v.X for v in self.v.values()])
