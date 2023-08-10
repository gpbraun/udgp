"""
Gabriel Braun, 2023

Este módulo implementa o modelo base para instâncias do problema uDGP usando a API gurobipy.
"""

from itertools import combinations, product

import gurobipy as gp
import numpy as np


class GPBaseModel(gp.Model):
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
        max_gap=1.0e-4,
        max_tol=1.0e-6,
        relaxed=False,
        previous_a: list | None = None,
        env=None,
    ):
        super(GPBaseModel, self).__init__("uDGP", env)

        # PARÂMETROS DA INSTÂNCIA
        self.ny = len(y_indices)
        self.nx = len(x_indices)
        self.m = len(dists)
        self.runtime = 0

        # CONJUNTOS
        ## Conjunto I
        self.Iy = set(y_indices)
        self.Ix = set(x_indices)
        self.I = self.Iy | self.Ix

        ## Conjunto IJ
        self.IJyx = set(product(self.Iy, self.Ix))
        self.IJxx = set(combinations(self.Ix, 2))
        self.IJ = self.IJyx | self.IJxx

        ## Conjunto K
        all_k = np.arange(self.m)
        self.K = {k for k in all_k if freqs[k] > 0}
        self.IJK = {(i, j, k) for (i, j), k in product(self.IJ, self.K)}

        # PARÂMETROS
        self.max_gap = max_gap
        self.max_tol = max_tol
        self.max_err = self.max_gap + self.max_tol
        self.d_min = dists.min() - self.max_err
        self.d_max = dists.max() + self.max_err

        self.dists = dists
        self.freqs = freqs
        self.y = {i: fixed_points[i] for i in y_indices}

        # VARIÁVEIS BASE
        ## Decisão: distância k é referente ao par de átomos i e j
        self.relaxed = relaxed
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

        # RESTRIÇÕES
        self.addConstrs(self.a.sum("*", "*", k) <= self.freqs[k] for k in self.K)
        self.addConstrs(self.a.sum(i, j, "*") == 1 for i, j in self.IJ)
        self.addConstrs(self.v[i, j] == self.x[i] - self.x[j] for i, j in self.IJxx)
        self.addConstrs(self.v[i, j] == self.y[i] - self.x[j] for i, j in self.IJyx)
        self.addConstrs(
            self.r[i, j] ** 2 == self.v[i, j] @ self.v[i, j] for i, j in self.IJ
        )

        # RESTRIÇÕES PARA SOLUÇÕES ANTERIORES
        previous_a = previous_a if previous_a is not None else []
        for a_ijk_indices in previous_a:
            self.addConstr(
                gp.quicksum(self.a[i, j, k] for i, j, k in a_ijk_indices)
                <= len(a_ijk_indices) - 1
            )

        self.update()

    def __setattr__(self, *args):
        try:
            return super(GPBaseModel, self).__setattr__(*args)
        except AttributeError:
            return super(gp.Model, self).__setattr__(*args)

    def solution_points(self):
        """
        Retorna (numpy.ndarray): pontos encontrados na solução do modelo.
        """
        return np.array([self.x[i].X for i in self.Ix])

    def solve(self, log=False):
        """
        Otimiza o modelo e atualiza a instância.

        Retorna: verdadeiro se uma solução foi encontrada
        """
        # PARÂMETROS DO SOLVER
        mip_gap = self.max_gap

        self.Params.LogToConsole = log
        self.Params.NonConvex = 2
        self.Params.MIPGap = mip_gap
        self.Params.IntFeasTol = self.max_tol
        self.Params.FeasibilityTol = self.max_tol
        self.Params.OptimalityTol = self.max_tol
        # self.Params.Cuts = 2

        # OTIMIZA
        super(GPBaseModel, self).optimize()
        self.runtime = self.Runtime

        if self.Status == gp.GRB.INFEASIBLE or self.SolCount == 0:
            print("Modelo inviável.")
            # self.status == "infeasible"
            return False

        return True
