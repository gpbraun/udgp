"""Gabriel Braun, 2023

Este módulo implementa a classe base para instâncias do problema uDGP.
"""

import numpy as np

from udgp.models import M1, M2, M1gp, M2gp
from udgp.utils import *

from .artificial_molecule import artificial_molecule_points
from .nanoparticles import c60, lj_cluster_points


class Instance:
    """
    Instância para o problema uDGP.
    """

    def __init__(
        self,
        n: int,
        dists: np.ndarray,
        freqs: np.ndarray | None = None,
        points: np.ndarray | None = None,
    ):
        if freqs is None:
            freqs = np.ones_like(dists, dtype=np.int64)

        self.n = n

        self.dists = dists
        self.freqs = freqs
        self.points = np.zeros((1, 3), dtype=np.float16)

        self.a_indices = np.empty((0, 3), dtype=np.int16)
        self.status = "ok"
        self.runtime = 0.0

        self.input_freqs = freqs
        self.input_points = points

    @property
    def m(self):
        """
        Retorna (int): número de distsâncias remanescentes.
        """
        return len(self.dists)

    @property
    def fixed_n(self):
        """
        Retorna (int): número de átomos da solução atual (fixados).
        """
        return len(self.points)

    @property
    def repeat_dists(self):
        """
        Retorna (numpy.ndarray): lista ordenada com repetição de distâncias remanescentes.
        """
        return np.repeat(self.dists, self.freqs)

    @property
    def input_repeat_dists(self):
        """
        Retorna (numpy.ndarray): lista ordenada com repetição de distâncias de entrada.
        """
        return np.repeat(self.dists, self.input_freqs)

    @property
    def solution_repeat_dists(self):
        """
        Retorna (numpy.ndarray): lista ordenada com repetição de distâncias já utilizadas.
        """
        return points_dists(self.points)

    def view(self, *args, **kwargs):
        """
        Retorna (py3Dmol.view): visualização da solução com py3Dmol.
        """
        return points_view(self.points, *args, **kwargs)

    def view_input(self, *args, **kwargs):
        """
        Retorna (py3Dmol.view): visualização da instância com py3Dmol.
        """
        if self.input_points is None:
            return

        return points_view(self.input_points, *args, **kwargs)

    def is_solved(self, threshold=1e-3):
        """
        Retorna (bool): verdadeiro se as distsâncias do input são as mesmas da da solução.
        """
        solution_repeat_dists = points_dists(self.points)

        if solution_repeat_dists.shape != self.input_repeat_dists.shape:
            return False

        var = np.var(solution_repeat_dists - self.input_repeat_dists)

        return var < threshold

    def reset(self, reset_runtime=False):
        """
        Reseta a instância para o estado inicial.
        """
        self.dists = self.dists.copy()
        self.freqs = self.input_freqs.copy()
        self.points = np.zeros((1, 3), dtype=np.float16)
        self.a_indices = np.empty((0, 3), dtype=np.int16)
        if reset_runtime:
            self.runtime = 0.0

    def remove_dists(self, dists: np.ndarray, indices: np.ndarray, threshold=0.05):
        """
        Remove uma lista distâncias com repetições da lista de distsâncias remanescentes.

        Parâmetros:
        - dists (numpy.ndarray): lista de distância com repetições para serem removidas
        - indices (numpy.ndarray): índices referentes às distâncias.

        Retorna (bool): verdadeiro se todas as distsâncias fornecidas estavam na lista de distsâncias remanescentes.
        """
        new_freqs = self.freqs.copy()

        for dist, (i, j) in zip(dists, indices):
            if dist == 0:
                return False

            errors = np.ma.array(np.abs(1 - self.dists / dist), mask=new_freqs == 0)
            k = errors.argmin()

            if errors[k] < threshold:
                new_freqs[k] -= 1
                self.a_indices = np.vstack((self.a_indices, [i, j, k]))
            else:
                print(f"distância {dist} não está na lista: {self.dists}")
                print(f"frequências: {new_freqs}")
                print(f"erro mínimo: {self.dists[k]} -> {errors[k]}")
                return False

        self.freqs = new_freqs

        return True

    def add_points(self, new_points: np.ndarray, threshold=0.05):
        """
        Adiciona (fixa) novas coordenadas à solução.

        Retorna (bool): verdadeiro se as distâncias entre os novos pontos e os pontos já fixados estavam na lista de distâncias remanescentes.
        """
        new_dists, new_indices = points_new_dists(
            new_points, self.points, return_indices=True
        )

        if self.remove_dists(new_dists, new_indices, threshold):
            self.points = np.r_[self.points, new_points]
            return True

        return False

    def reset_with_core(self, core_type: str, n=5):
        """
        Reinicia a instância com um core de molécula artificial de n átomos como solução inicial.
        """
        core_found = False

        while not core_found:
            self.reset()

            if core_type == "mock":
                rng = np.random.default_rng()
                y_indices = rng.choice(self.n, n, replace=False)
                core_points = self.input_points[y_indices]

            elif core_type == "artificial":
                core_points = artificial_molecule_points(n)

            core_dists, core_indices = points_dists(core_points, return_indices=True)
            core_found = self.remove_dists(core_dists, core_indices)

        self.points = core_points

    def solve_heuristic(self):
        """
        Usa uma heurística baseada no método TRIBOND para resolver a instância.
        """
        while not self.is_solved():
            self.reset()
            self.solve("M2", nx=4)
            return

    def solve_step(
        self,
        model_name,
        nx: int | None = None,
        ny: int | None = None,
        max_gap: float = 5e-3,
        log=False,
        previous_a: list | None = None,
    ):
        """
        Resolve a instância.
        """
        ny = self.fixed_n if ny is None else ny
        nx = self.n - self.fixed_n if nx is None else nx

        rng = np.random.default_rng()
        y_indices = np.sort(rng.choice(self.fixed_n, ny, replace=False))
        x_indices = np.arange(self.fixed_n, nx + self.fixed_n)

        relaxed = "r" in model_name.lower()
        gurobipy = "gp" in model_name.lower()

        if "M1" in model_name:
            model = M1gp if gurobipy else M1
        elif "M2" in model_name:
            model = M2gp if gurobipy else M2

        m = model(
            x_indices,
            y_indices,
            self.dists,
            self.freqs,
            self.points,
            max_gap=max_gap,
            relaxed=relaxed,
            previous_a=previous_a,
        )

        solved = m.solve(log=log)

        self.runtime += m.runtime

        if not solved:
            return False

        # ATUALIZA A INSTÂNCIA
        new_points = m.solution_points()
        if not self.add_points(new_points, 0.1):
            return False
        else:
            return True

    @classmethod
    def from_points(cls, points, freq=True):
        """
        Retorna (Instance): instância referente às coordenadas fornecidas.
        """
        dists = points_dists(points)
        freqs = None

        if freq:
            dists, freqs = np.unique(dists, return_counts=True)

        return cls(points.shape[0], dists, freqs, points)

    @classmethod
    def artificial_molecule(cls, n: int, seed: int = None, freq=True):
        """
        Retorna (Instance): instância de molécula artificial com n átomos.

        Referência: Lavor, C. (2006) https://doi.org/10.1007/0-387-30528-9_14
        """
        points = artificial_molecule_points(n, seed)

        return cls.from_points(points, freq)

    @classmethod
    def c60(cls, freq=True):
        """
        Retorna (Instance): instância de cluster de Lennard-Jones com n (entre 3 e 150) átomos.

        Referência: https://webbook.nist.gov/cgi/inchi?ID=C99685968&Mask=20
        """
        points = c60()

        return cls.from_points(points, freq)

    @classmethod
    def lj_cluster(cls, n, freq=True):
        """
        Retorna (Instance): instância de cluster de Lennard-Jones com n (entre 3 e 150) átomos.

        Referência: https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html
        """
        points = lj_cluster_points(n)

        return cls.from_points(points, freq)
