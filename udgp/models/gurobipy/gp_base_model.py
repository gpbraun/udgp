"""
Gabriel Braun, 2023

Este módulo implementa o modelo base para instâncias do problema uDGP usando a API gurobipy.
"""

from itertools import chain, combinations, product

import gurobipy as gp
import numpy as np

from udgp.solvers import get_config


class gpBaseModel(gp.Model):
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
        env=None,
    ):
        env = env or gp.Env(empty=True)
        env.setParam("LogToConsole", 0)
        env.start()

        super(gpBaseModel, self).__init__("uDGP", env)
        self.name = "Base"

        # PARÂMETROS DA INSTÂNCIA
        self.ny = len(y_indices)
        self.nx = len(x_indices)
        self.m = len(dists)
        self.runtime = 0

        # CONJUNTOS
        ## Conjunto I
        self.Iy = y_indices
        self.Ix = x_indices
        self.I = list(chain(self.Iy, self.Ix))

        ## Conjunto IJ
        self.IJyx = list(product(self.Iy, self.Ix))
        self.IJxx = list(combinations(self.Ix, 2))
        self.IJ = list(chain(self.IJyx, self.IJxx))

        ## Conjunto K
        all_k = list(range(self.m))
        self.K = [k for k in all_k if freqs[k] != 0]
        self.IJK = [(i, j, k) for (i, j), k in product(self.IJ, self.K)]

        # PARÂMETROS
        self.d_min = dists.min()
        self.d_max = dists.max()

        self.dists = dists
        self.freqs = freqs
        self.y = {i: fixed_points[i] for i in y_indices}

        # VARIÁVEIS BASE
        self.relaxed = relaxed
        ## Decisão: distância k é referente ao par de átomos i e j
        if self.relaxed:
            self.a = self.addVars(
                self.IJK,
                name="a",
                vtype=gp.GRB.CONTINUOUS,
                lb=0,
                ub=1,
            )
        else:
            self.a = self.addVars(
                self.IJK,
                name="a",
                vtype=gp.GRB.BINARY,
            )
        ## Coordenadas do ponto i
        self.x = {
            i: self.addMVar(
                3,
                name=f"x[{i}]",
                vtype=gp.GRB.CONTINUOUS,
                lb=-gp.GRB.INFINITY,
                ub=gp.GRB.INFINITY,
            )
            for i in self.Ix
        }
        ## Vetor distância entre os átomos i e j
        self.v = {
            ij: self.addMVar(
                3,
                name=f"v[{ij}]",
                vtype=gp.GRB.CONTINUOUS,
                lb=-gp.GRB.INFINITY,
                ub=gp.GRB.INFINITY,
            )
            for ij in self.IJ
        }
        ## Distância entre os átomos i e j (norma de v)
        self.r = self.addVars(
            self.IJ,
            name="r",
            vtype=gp.GRB.CONTINUOUS,
            lb=self.d_min,
            ub=self.d_max,
        )

        # RESTRIÇÕES BASE
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
            self.r[i, j] ** 2 == self.v[i, j] @ self.v[i, j] for i, j in self.IJ
        )

        # RESTRIÇÕES PARA SOLUÇÕES ANTERIORES
        previous_a = previous_a if previous_a is not None else []

        self._constr_previous_a = self.addConstrs(
            gp.quicksum(self.a[i, j, k] for i, j, k in a_ijk_indices)
            <= len(a_ijk_indices) - 1
            for a_ijk_indices in previous_a
        )

        self.update()

    def __setattr__(self, *args):
        try:
            return super(gpBaseModel, self).__setattr__(*args)
        except AttributeError:
            return super(gp.Model, self).__setattr__(*args)

    def solution_points(self) -> np.ndarray:
        """
        Retorna (numpy.ndarray): pontos encontrados na solução do modelo.
        """
        return np.array([self.x[i].X for i in self.Ix])

    def solve(
        self,
        *,
        config: dict | None = None,
        stage: str | None = None,
    ) -> bool:
        """
        Otimiza o modelo e atualiza a instância.

        Retorna: verdadeiro se uma solução foi encontrada
        """
        config = get_config(
            solver="gurobi",
            model=self.name,
            stage=stage,
            overrides=config,
        )
        for k, v in config.items():
            self.setParam(k, v)
        # OTIMIZA
        super(gpBaseModel, self).optimize()

        if self.Status == gp.GRB.INFEASIBLE or self.SolCount == 0:
            # self.status == "infeasible"
            return False

        return True
