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
        max_gap=1e-2,
        max_tol=1e-3,
        env=None,
    ):
        super(BaseModel, self).__init__("uDGP", env)

        self.Params.LogToConsole = False
        self.Params.MIPGap = max_gap
        self.Params.NonConvex = 2

        self.Params.IntFeasTol = max_tol
        self.Params.FeasibilityTol = max_tol
        self.Params.OptimalityTol = max_tol

        self.instance = instance
        self.m = instance.m

        ## ÁTOMOS NOVOS (x)
        if nx is None or nx > instance.n - instance.fixed_n:
            nx = instance.n - instance.fixed_n
        self.x_n = nx

        self.x_index = np.arange(self.instance.fixed_n, nx + self.instance.fixed_n)

        ## ÁTOMOS FIXADOS (y)
        if ny is None or ny > instance.fixed_n:
            ny = instance.fixed_n
        self.y_n = ny

        rng = np.random.default_rng()
        self.y_index = rng.choice(instance.fixed_n, ny, replace=False)
        self.y = [self.y_index]

        # VARIÁVEIS
        ## Decisão: distância k é referente ao par de átomos i e j
        self.a = self.addVars(
            self.ijk_index(),
            name="a",
            vtype=GRB.BINARY,
        )
        ## Coordenadas do átomo i
        self.x = {
            i: self.addMVar(
                3,
                name=f"x[{i}]",
                vtype=GRB.CONTINUOUS,
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
            )
            for i in self.x_index
        }
        self.y = {i: instance.coords[i] for i in self.y_index}
        ## Vetor distância entre os átomos i e j
        self.v = {
            ij: self.addMVar(
                3,
                name=f"v[{ij}]",
                vtype=GRB.CONTINUOUS,
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
            )
            for ij in self.ij_index()
        }
        ## Distância entre os átomos i e j (norma de v)
        self.r = self.addVars(
            self.ij_index(),
            name="r",
            vtype=GRB.CONTINUOUS,
            lb=0.5,
            ub=GRB.INFINITY,
        )

        # RESTRIÇÕES
        self.addConstrs(
            self.a.sum("*", "*", k) <= self.instance.freq[k] for k in self.k_index()
        )
        self.addConstrs(self.a.sum(i, j, "*") == 1 for i, j in self.ij_index())
        self.addConstrs(
            self.v[i, j] == self.x[i] - self.x[j]
            for i, j in self.ij_index(xx=True, xy=False)
        )
        self.addConstrs(
            self.v[i, j] == self.y[i] - self.x[j]
            for i, j in self.ij_index(xx=False, xy=True)
        )
        self.addConstrs(
            self.r[i, j] ** 2 == self.v[i, j] @ self.v[i, j] for i, j in self.ij_index()
        )

        # ÍNDICES PROIBIDOS
        if previous_a is not None:
            for a_ijk in previous_a:
                self.addConstr(
                    gp.quicksum(self.a[ijk] for ijk in a_ijk) <= len(a_ijk) - 1
                )

        self.update()

    def __setattr__(self, *args):
        try:
            return super(BaseModel, self).__setattr__(*args)
        except AttributeError:
            return super(gp.Model, self).__setattr__(*args)

    def k_index(self) -> Iterator[int]:
        """
        Retorna: índices k.
        """
        for k in np.arange(self.m):
            if self.instance.freq[k] > 0:
                yield k

    def i_index(self) -> Iterator[int]:
        """
        Retorna: índices i.
        """
        # fixado
        for i in self.y_index:
            yield i
        # interno
        for i in self.x_index:
            yield i

    def ij_index(self, xx=True, xy=True) -> Iterator[int]:
        """
        Retorna: índices i, j.
        """
        # fixado, interno
        if xy:
            for i in self.y_index:
                for j in self.x_index:
                    yield i, j
        # interno, interno
        if xx:
            for i in self.x_index:
                for j in self.x_index:
                    if i < j:
                        yield i, j

    def ijk_index(self) -> Iterator[int]:
        """
        Retorna: índices i, j, k.
        """
        for i, j in self.ij_index():
            for k in self.k_index():
                yield i, j, k

    def a_ijk_index(self) -> Iterator[int]:
        """
        Retorna: índices i, j, k dos valores de a selecionados.
        """
        for i, j, k in self.ijk_index():
            if self.a[i, j, k].X == 1:
                yield i, j, k

    def optimize(self, log=False):
        """
        Otimiza o modelo e atualiza a instância.
        """
        self.Params.LogToConsole = log
        super(BaseModel, self).optimize()

        if self.Status == GRB.INFEASIBLE:
            print("Modelo inviável.")
            return

        if self.SolCount == 0:
            return

        new_coords = np.array([self.x[i].X for i in self.x_index])
        if not self.instance.add_coords(new_coords):
            self.Status = GRB.INTERRUPTED
