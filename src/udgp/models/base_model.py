"""Gabriel Braun, 2023

Este módulo implementa o modelo base para instâncias do problema uDGP.
"""

from collections.abc import Iterator
from copy import deepcopy

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from udgp.instances.instance import Instance


class BaseModel(gp.Model):
    """Modelo base para o uDGP."""

    def __init__(
        self,
        instance: Instance,
        n: int | None = None,
        previous_a=None,
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

        self.instance = deepcopy(instance)
        self.n = n if n is not None else instance.n - instance.fixed_n
        self.m = self.instance.m

        ## ÁTOMOS FIXADOS
        self.fixed_coords = self.instance.get_random_coords(n=10)
        fixed_num = self.fixed_coords.shape[0]

        # VARIÁVEIS
        ## Decisão: distância k é referente ao par de átomos i e j
        self.a = self.addVars(
            self.ijk_values(),
            name="a",
            vtype=GRB.BINARY,
        )
        ## Coordenadas do átomo i
        self.x = self.addMVar(
            (self.n + fixed_num, 3),
            name="x",
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
        )
        ## Vetor distância entre os átomos i e j
        self.v = {
            ij: self.addMVar(
                3,
                name=f"v[{ij}]",
                vtype=GRB.CONTINUOUS,
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
            )
            for ij in self.ij_values()
        }
        ## Distância entre os átomos i e j (norma de v)
        self.r = self.addVars(
            self.ij_values(),
            name="r",
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=GRB.INFINITY,
        )

        # RESTRIÇÕES
        self.addConstrs(
            self.a.sum("*", "*", k) <= self.instance.freq[k] for k in self.k_values()
        )
        self.addConstrs(self.a.sum(i, j, "*") == 1 for i, j in self.ij_values())
        self.addConstrs(
            self.v[i, j] == self.x[i] - self.x[j] for i, j in self.ij_values()
        )
        self.addConstrs(
            self.r[i, j] ** 2 == self.v[i, j] @ self.v[i, j]
            for i, j in self.ij_values()
        )
        self.addConstr(self.x[:fixed_num] == self.fixed_coords)

        # CORE
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

    def k_values(self) -> Iterator[int]:
        """Retorna: índices k."""
        for k in np.arange(self.m):
            yield k

    def ij_values(self) -> Iterator[int]:
        """Retorna: índices i, j."""
        n_fixed = self.fixed_coords.shape[0]
        # fixado, interno
        for i in np.arange(n_fixed):
            for j in np.arange(n_fixed, n_fixed + self.n):
                yield i, j
        # interno, interno
        for i in np.arange(n_fixed, self.n + n_fixed - 1):
            for j in np.arange(n_fixed + i, n_fixed + self.n):
                yield i, j

    def ijk_values(self) -> Iterator[int]:
        """Retorna: índices i, j, k."""
        for i, j in self.ij_values():
            for k in self.k_values():
                yield i, j, k

    def a_ijk_values(self) -> Iterator[int]:
        """Retorna: índices i, j, k dos valores de a selecionados."""
        for i, j, k in self.ijk_values():
            if self.a[i, j, k].X == 1:
                yield i, j, k

    def optimize(self, log=False):
        """Otimiza o modelo e atualiza a instância."""
        self.Params.LogToConsole = log

        super(BaseModel, self).optimize()

        if self.Status == GRB.INFEASIBLE:
            print("Modelo inviável.")
            return

        if self.SolCount == 0:
            return

        for i, j, k in self.a_ijk_values():
            self.instance.freq[k] -= 1
        self.instance.remove_zero_freq()

        self.instance.add_coords(self.x.X)
