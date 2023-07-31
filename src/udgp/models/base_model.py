"""Gabriel Braun, 2023

Este módulo implementa o modelo base para instâncias do problema uDGP.
"""

from collections.abc import Iterator

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from udgp.instances.instance import Instance


class BaseModel(gp.Model):
    """
    Modelo base para o uDGP.
    """

    def __init__(
        self,
        instance: Instance,
        nx: int | None = None,
        ny: int | None = None,
        previous_a: list | None = None,
        max_dist: float | None = None,
        max_gap=1e-2,
        max_tol=1e-3,
        core=False,
        env=None,
    ):
        super(BaseModel, self).__init__("uDGP", env)

        self.Params.LogToConsole = False
        self.Params.NonConvex = 2
        # self.Params.SolutionLimit = 1
        self.Params.MIPGap = max_gap

        self.Params.IntFeasTol = max_tol
        self.Params.FeasibilityTol = max_tol
        self.Params.OptimalityTol = max_tol

        self.instance = instance
        self.m = instance.m
        self.max_dist = max_dist

        ## ÁTOMOS NOVOS (x)
        if nx is None or nx > instance.n - instance.fixed_n:
            nx = instance.n - instance.fixed_n
        self.x_n = nx

        self.x_indices = np.arange(self.instance.fixed_n, nx + self.instance.fixed_n)

        ## ÁTOMOS FIXADOS (y)
        if ny is None or ny > instance.fixed_n:
            ny = instance.fixed_n
        self.y_n = ny

        rng = np.random.default_rng()
        self.y_indices = rng.choice(instance.fixed_n, ny, replace=False)
        self.y = instance.points[self.y_indices]

        # VARIÁVEIS
        ## Decisão: distância k é referente ao par de átomos i e j
        self.a = self.addVars(
            self.ijk_indices(),
            name="a",
            vtype=GRB.BINARY,
        )
        ## pointenadas do átomo i
        self.x = {
            i: self.addMVar(
                3,
                name=f"x[{i}]",
                vtype=GRB.CONTINUOUS,
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
            )
            for i in self.x_indices
        }
        self.y = {i: instance.points[i] for i in self.y_indices}
        ## Vetor distância entre os átomos i e j
        self.v = {
            ij: self.addMVar(
                3,
                name=f"v[{ij}]",
                vtype=GRB.CONTINUOUS,
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
            )
            for ij in self.ij_indices()
        }
        ## Distância entre os átomos i e j (norma de v)
        self.r = self.addVars(
            self.ij_indices(),
            name="r",
            vtype=GRB.CONTINUOUS,
            lb=0.5,
            ub=GRB.INFINITY,
        )

        # RESTRIÇÕES
        self.addConstrs(
            self.a.sum("*", "*", k) <= self.instance.freqs[k] for k in self.k_indices()
        )
        self.addConstrs(self.a.sum(i, j, "*") == 1 for i, j in self.ij_indices())
        self.addConstrs(
            self.v[i, j] == self.x[i] - self.x[j]
            for i, j in self.ij_indices(xx=True, xy=False)
        )
        self.addConstrs(
            self.v[i, j] == self.y[i] - self.x[j]
            for i, j in self.ij_indices(xx=False, xy=True)
        )
        self.addConstrs(
            self.r[i, j] ** 2 == self.v[i, j] @ self.v[i, j]
            for i, j in self.ij_indices()
        )

        if core:
            self.addConstr(self.a.sum("*", "*", 0) >= min(self.instance.freqs[0], 4))

        # ÍNDICES PROIBIDOS
        if previous_a is not None:
            for a_ijk in previous_a:
                try:
                    self.addConstr(
                        gp.quicksum(self.a[ijk] for ijk in a_ijk) <= len(a_ijk) - 1
                    )
                except:
                    pass

        self.update()

    def __setattr__(self, *args):
        try:
            return super(BaseModel, self).__setattr__(*args)
        except AttributeError:
            return super(gp.Model, self).__setattr__(*args)

    def k_indices(self):
        """
        Retorna: índices k.
        """
        for k in np.arange(self.m):
            if self.instance.freqs[k] > 0 and self.instance.dists[k] < self.max_dist:
                yield k

    def i_indices(self):
        """
        Retorna: índices i.
        """
        # fixado
        for i in self.y_indices:
            yield i
        # interno
        for i in self.x_indices:
            yield i

    def ij_indices(self, xx=True, xy=True):
        """
        Retorna: índices i, j.
        """
        # fixado, interno
        if xy:
            for i in self.y_indices:
                for j in self.x_indices:
                    yield i, j
        # interno, interno
        if xx:
            for i in self.x_indices:
                for j in self.x_indices:
                    if i < j:
                        yield i, j

    def ijk_indices(self):
        """
        Retorna: índices i, j, k.
        """
        for i, j in self.ij_indices():
            for k in self.k_indices():
                yield i, j, k

    def a_ijk_indices(self):
        """
        Retorna: índices i, j, k dos valores de a selecionados.
        """
        for i, j, k in self.ijk_indices():
            if self.a[i, j, k].X == 1:
                yield i, j, k

    def optimize(self, log=False):
        """
        Otimiza o modelo e atualiza a instância.

        Retorna: verdadeiro se uma solução foi encontrada
        """
        self.Params.LogToConsole = log
        super(BaseModel, self).optimize()

        if self.Status == GRB.INFEASIBLE:
            print("Modelo inviável.")
            return False

        if self.SolCount == 0:
            return False

        new_points = np.array([self.x[i].X for i in self.x_indices])
        if not self.instance.add_points(new_points):
            return False
        else:
            return True
