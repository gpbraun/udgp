"""
gp_base_model.py
"""

import logging
from itertools import chain, combinations, product
from typing import Any

import gurobipy as gp
import numpy as np

from udgp.solver import get_solver_params

logger = logging.getLogger("udgp")


class gpBaseModel(gp.Model):
    """
    uDGP base model.
    """

    NAME = "Base"
    PARAMS = {
        "Relax": 0,
        "Lambda": 1,
    }

    def __setattr__(self, *args):
        try:
            return super(gpBaseModel, self).__setattr__(*args)
        except AttributeError:
            return super(gp.Model, self).__setattr__(*args)

    def _add_core_vars(self):
        """
        Add the variables to the models.
        """
        # Decisão: distância k é referente ao par de átomos i e j
        self.a = self.addVars(
            self.IJK,
            name="a",
            vtype=gp.GRB.BINARY,
        )
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
            lb=self.d_min,
            ub=self.d_max,
        )

    def _add_core_constrs(self):
        """
        Adds the constraints to the model
        """
        self._constr_a1 = self.addConstrs(
            self.a.sum("*", "*", k) <= self.freqs[k] for k in self.K
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
            self.r[i, j] == self.v[i, j] @ self.v[i, j] for i, j in self.IJ
        )

    def add_previous_solution_constrs(self, previous_a: list[tuple[int]]):
        """
        Adds constraints to prevent previous solutions.
        """
        self._constr_previous_a = self.addConstrs(
            gp.quicksum(self.a[ijk] for ijk in a_ijk_idx) <= len(a_ijk_idx) - 1
            for a_ijk_idx in previous_a
        )

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
        env=None,
    ):
        if not env:
            env = gp.Env(empty=True)
            env.setParam("LogToConsole", 0)
            env.start()

        super(gpBaseModel, self).__init__(f"uDGP_{self.NAME}", env)

        self.set_model_params(overrides=model_params)

        self.total_runtime = 0
        self.total_work = 0

        # INSTANCE ATTRS
        self.m = len(dists)
        self.nx = len(x_indices)
        self.ny = len(y_indices)

        # INSTANCE SETS
        ## I Sets
        self.Iy = gp.tuplelist(y_indices.tolist())
        self.Ix = gp.tuplelist(x_indices.tolist())
        self.I = gp.tuplelist(chain(self.Iy, self.Ix))
        ## IJ Sets
        self.IJyx = gp.tuplelist(product(self.Iy, self.Ix))
        self.IJxx = gp.tuplelist(combinations(self.Ix, 2))
        self.IJ = gp.tuplelist(chain(self.IJyx, self.IJxx))
        ## K Set
        self.K = gp.tuplelist([k for k in range(self.m) if freqs[k] != 0])
        self.IJK = gp.tuplelist([(i, j, k) for (i, j), k in product(self.IJ, self.K)])

        # OTHER INSTANCE ATTRS
        self.y = gp.tupledict((i, fixed_points[i]) for i in y_indices)
        self.dists = dists**2
        self.freqs = freqs
        self.d_min = self.dists.min()
        self.d_max = self.dists.max()

        # VARIABLES AND CONSTRAINTS
        self._add_core_vars()
        self._add_core_constrs()
        self.model_post_init()

        self.update()

    @property
    def assignments(self) -> dict[tuple[int], int]:
        """
        Retorna (list[tuple]): indices que correspondem aos valores de a unitários.
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

    def relax(self):
        """
        Relax `a` integer variables.
        """
        self.a.vtype = gp.GRB.CONTINUOUS
        self.a.lb = 0.0
        self.a.ub = 1.0

        lbd = self.model_params["Lambda"]

        self.objective += lbd * (
            len(self.IJ) - gp.quicksum(a**2 for a in self.a.values())
        )

    def model_pre_solve(self):
        """
        Run *before* the solve() routine.

        Subclasses add their own logic here and should call super().
        """
        if self.model_params["Relax"]:
            self.relax()

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

    def solve(
        self,
        *,
        stage: str | None = None,
        solver_params: dict | None = None,
    ):
        """
        Resolve o modelo.

        Retorna: verdadeiro se uma solução foi encontrada
        """
        self.model_pre_solve()

        self._set_solver_params(stage=stage, overrides=solver_params)
        self.optimize()

        if self.SolCount == 0:
            return False

        self.total_runtime += self.Runtime
        self.total_work += self.Work

        return True
