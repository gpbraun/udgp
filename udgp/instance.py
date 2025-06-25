"""Gabriel Braun, 2023

Este módulo implementa a classe base para instâncias do problema uDGP.
"""

import logging

import numpy as np

from udgp.instances import *
from udgp.models import get_model
from udgp.utils.points import points_dists, points_view

logger = logging.getLogger(__name__)


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
            freqs = np.ones_like(dists, dtype=np.int16)

        self.n = n

        self.dists = dists
        self.freqs = freqs
        self.points = np.zeros((1, 3), dtype=np.float16)

        self.a_indices = []
        self.status = "ok"
        self.runtime = 0.0
        self.work = 0.0

        self.input_freqs = freqs
        self.input_points = points

    @property
    def m(self):
        """
        Retorna (int): número de distsâncias remanescentes.
        """
        return len(self.dists)

    @property
    def n_fixed(self):
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
        Retorna (py3Dmol.view): visualização da solução encontrada com py3Dmol.
        """
        return points_view(self.points, *args, **kwargs)

    def view_input(self, *args, **kwargs):
        """
        Retorna (py3Dmol.view): visualização da instância original com py3Dmol.
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

    def reset(self, reset_runtime=True):
        """
        Reseta a instância para o estado inicial.
        """
        self.dists = self.dists.copy()
        self.freqs = self.input_freqs.copy()
        self.points = np.zeros((1, 3), dtype=np.float16)
        self.a_indices = []
        if reset_runtime:
            self.runtime = 0.0
            self.work = 0.0

    def reset_with_core(
        self,
        core_type: str,
        n: int = 5,
    ) -> None:
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

    def solve(
        self,
        model_name: str,
        *,
        nx: int | None = None,
        ny: int | None = None,
        relax_a: bool = False,
        previous_a: list | None = None,
        model_params: dict | None = None,
        solver_params: dict | None = None,
        backend="gurobipy",
    ) -> bool:
        """
        Resolve a instância.
        """
        ny = ny or self.n_fixed
        nx = nx or self.n - self.n_fixed

        rng = np.random.default_rng()
        y_indices = np.sort(rng.choice(self.n_fixed, ny, replace=False))
        x_indices = np.arange(self.n_fixed, nx + self.n_fixed)

        model = get_model(
            model_name,
            backend=backend,
            x_indices=x_indices,
            y_indices=y_indices,
            dists=self.dists,
            freqs=self.freqs,
            fixed_points=self.points,
            model_params=model_params,
            previous_a=previous_a,
        )
        if relax_a:
            model.relax_a()

        solve_ok = model.solve(solver_params=solver_params)

        self.runtime += model.total_runtime
        self.work += model.total_work

        if not solve_ok:
            return False

        # UPDATE INSTANCE
        self.points = np.r_[self.points, model.sol_x_array]

        for i, j, k in model.sol_a_indices:
            error = abs(np.linalg.norm(model.sol_v[i, j]) - self.dists[k])
            if error > 1e-2:
                logger.info(f"ERRO NA DISTÂNCIA ({i}, {j}): {error}")

            self.a_indices.append((i, j, k))
            self.freqs[k] -= 1

        return True

    def solve_heuristic(self):
        """
        Usa uma heurística baseada no método TRIBOND para resolver a instância.
        """
        while not self.is_solved():
            self.reset()
            self.solve("M2", nx=4)
            return

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
        Retorna (Instance): instância de molécula artificial com `n` átomos.

        Referência: Lavor, C. (2006) https://doi.org/10.1007/0-387-30528-9_14
        """
        points = artificial_molecule_points(n, seed)

        return cls.from_points(points, freq)

    @classmethod
    def lj_cluster(cls, n, freq=True):
        """
        Retorna (Instance): instância de cluster de Lennard-Jones com `n` (entre 3 e 150) átomos.

        Referência: https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html
        """
        points = lj_cluster_points(n)

        return cls.from_points(points, freq)

    @classmethod
    def c60(cls, freq=True):
        """
        Retorna (Instance): instância de cluster de Lennard-Jones com `n` (entre 3 e 150) átomos.

        Referência: https://webbook.nist.gov/cgi/inchi?ID=C99685968&Mask=20
        """
        points = c60_points()

        return cls.from_points(points, freq)
