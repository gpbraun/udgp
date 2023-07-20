"""Gabriel Braun, 2023

Este módulo implementa o modelo base para instâncias do problema uDGP.
"""

from collections.abc import Iterator

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from udgp.instances.base_instance import Instance


class BaseModel(gp.Model):
    """Modelo base para o uDGP."""

    def __init__(
        self,
        instance: Instance,
        name="uDGP",
        log=True,
        max_gap=1e-4,
        env=None,
    ):
        super(BaseModel, self).__init__(name, env)

        self.setParam("LogToConsole", log)
        self.setParam("MIPGap", max_gap)
        self.setParam("NonConvex", 2)

        self.max_gap = max_gap

        self.instance = instance
        self.n = instance.n
        self.m = instance.m

        # VARIÁVEIS
        ## Decisão: distância k é referente ao par de átomos i e j
        self.a = self.addVars(
            self.ijk_values(),
            name="a",
            vtype=GRB.BINARY,
        )
        ## Coordenadas do átomo i
        self.x = self.addMVar(
            (self.n, 3),
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
        self.addConstrs(self.a.sum("*", "*", k) <= 1 for k in self.k_values())
        self.addConstrs(self.a.sum(i, j, "*") == 1 for i, j in self.ij_values())
        self.addConstrs(
            self.v[i, j] == self.x[i] - self.x[j] for i, j in self.ij_values()
        )
        self.addConstrs(
            self.r[i, j] ** 2 == self.v[i, j] @ self.v[i, j]
            for i, j in self.ij_values()
        )
        self.addConstr(self.x[0] == np.zeros(3))

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
        for i, j in np.transpose(np.triu_indices(self.n, 1)):
            yield i, j

    def ijk_values(self) -> Iterator[int]:
        """Retorna: índices i, j, k."""
        for i, j in self.ij_values():
            for k in self.k_values():
                yield i, j, k

    def solution(self) -> Instance:
        """Retorna: solução do modelo."""
        if self.SolCount == 0:
            return None

        return Instance(
            coords=self.x.X,
            distances=np.array([self.r[ij].X for ij in self.ij_values()]).sort(),
        )

    def solution_is_isomorphic(self) -> bool:
        """Retorna: verdadeiro se a entrada e a solução representam a mesma molécula."""
        if self.SolCount == 0:
            return False

        return self.solution().is_isomorphic(self.instance)
