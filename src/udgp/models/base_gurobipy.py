"""Gabriel Braun, 2023

Este módulo implementa o modelo base para instâncias do problema uDGP usando a API gurobipy.
"""

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from udgp.instances.instance import Instance


class BaseModelGurobipy(gp.Model):
    """
    Modelo base para o uDGP.
    """

    def __init__(
        self,
        instance: Instance,
        nx: int | None = None,
        ny: int | None = None,
        max_gap=5e-3,
        max_tol=1e-4,
        relaxed=False,
        env=None,
    ):
        super(BaseModelGurobipy, self).__init__("uDGP", env)

        # PARÂMETROS DA INSTÂNCIA
        self.instance = instance
        self.m = instance.m

        # ÍNDICES
        # pontos novos (x)
        if nx is None or nx > instance.n - instance.fixed_n:
            nx = instance.n - instance.fixed_n
        self.x_n = nx

        self.x_indices = np.arange(self.instance.fixed_n, nx + self.instance.fixed_n)

        # pontos fixados (y)
        if ny is None or ny > instance.fixed_n:
            ny = instance.fixed_n
        self.y_n = ny

        rng = np.random.default_rng()
        self.y_indices = rng.choice(instance.fixed_n, ny, replace=False)

        self.y = instance.points[self.y_indices]

        # PARÂMETROS DO SOLVER (GUROBI)
        self.Params.LogToConsole = False
        self.Params.NonConvex = 2

        self.max_gap = max_gap
        self.max_tol = max_tol

        self.Params.MIPGap = len(list(self.ij_indices())) * max_gap
        self.Params.IntFeasTol = max_tol
        self.Params.FeasibilityTol = max_tol
        self.Params.OptimalityTol = max_tol

        # PARÂMETROS
        ## coordenadas dos pontos fixados
        self.y = {i: instance.points[i] for i in self.y_indices}
        ## maior distância
        self.d_max = self.instance.dists[list(self.k_indices())].max()

        # VARIÁVEIS
        ## Decisão: distância k é referente ao par de átomos i e j
        self.relaxed = relaxed
        if self.relaxed:
            self.a = self.addVars(
                self.ijk_indices(),
                name="a",
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=1,
            )
        else:
            self.a = self.addVars(
                self.ijk_indices(),
                name="a",
                vtype=GRB.BINARY,
            )
        ## coordenadas do ponto i
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
            ub=self.d_max,
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

        self.update()

    def __setattr__(self, *args):
        try:
            return super(BaseModelGurobipy, self).__setattr__(*args)
        except AttributeError:
            return super(gp.Model, self).__setattr__(*args)

    def k_indices(self):
        """
        Retorna: índices k.
        """
        for k in np.arange(self.m):
            if self.instance.freqs[k] > 0:
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
        super(BaseModelGurobipy, self).optimize()

        if self.Status == GRB.INFEASIBLE:
            print("Modelo inviável.")
            return False

        if self.SolCount == 0:
            return False

        new_points = np.array([self.x[i].X for i in self.x_indices])
        if not self.instance.add_points(new_points, 2 * self.max_gap):
            return False
        else:
            return True
