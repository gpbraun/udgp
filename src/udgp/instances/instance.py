"""Gabriel Braun, 2023

Este módulo implementa a classe base para instâncias do problema uDGP.
"""


import numpy as np
from py3Dmol import view

import udgp.utils as utils

from .artificial_molecule import artificial_molecule_coords

START_COORDS = np.array([[0.0, 0.0, 0.0]])


class Instance:
    """
    Instância para o problema uDGP.
    """

    def __init__(
        self,
        n: int,
        dist: np.ndarray,
        freq: np.ndarray | None = None,
        coords: np.ndarray | None = None,
    ):
        if freq is None:
            freq = np.ones_like(dist, dtype=np.int64)

        self.n = n
        self.dist = dist
        self.freq = freq
        self.coords = START_COORDS.copy()
        self.input_dist = dist
        self.input_freq = freq
        self.input_coords = coords

    @property
    def m(self) -> int:
        """
        Retorna: número de distâncias remanescentes.
        """
        return self.dist.shape[0]

    @property
    def fixed_n(self) -> int:
        """
        Retorna: número de átomos da solução atual (fixados).
        """
        return self.coords.shape[0]

    @property
    def all_dist(self) -> np.ndarray:
        """
        Retorna: lista completa (ordenada) de distâncias remanescentes.
        """
        return np.repeat(self.dist, self.freq)

    @property
    def input_all_dist(self) -> np.ndarray:
        """
        Retorna: lista completa (ordenada) de distâncias de entrada.
        """
        return np.repeat(self.input_dist, self.input_freq)

    def view(self) -> view:
        """
        Retorna: visualização da solução encontrada com py3Dmol.
        """
        return utils.coords_to_view(self.coords)

    def view_input(self) -> view:
        """
        Retorna: visualização da instância com py3Dmol.
        """
        if self.input_coords is None:
            return

        return utils.coords_to_view(self.input_coords)

    def is_solved(self, threshold=1e-3) -> bool:
        """
        Retorna: verdadeiro se as distâncias do input são as mesmas da da solução.

        Referência: LIGA
        """
        solution_all_dist = utils.coords_to_dist(self.coords)

        if solution_all_dist.shape != self.input_all_dist.shape:
            return False

        var = np.var(solution_all_dist - self.input_all_dist)

        return var < threshold

    def get_random_coords(self, n=4) -> np.ndarray:
        """
        Retorna: n coordenadas já fixadas escolhidas aleatoriamente.
        """
        if n >= self.coords.shape[0]:
            return self.coords

        sample_coords = utils.split_coords(self.coords, n)[1]
        return sample_coords

    def reset(self) -> None:
        """
        Reseta a instância para o estado inicial.
        """
        self.dist = self.input_dist.copy()
        self.freq = self.input_freq.copy()
        self.coords = START_COORDS.copy()

    def remove_zero_freq(self):
        """
        Remove as distâncias cuja frequência é 0.
        """
        self.dist = self.dist[self.freq != 0]
        self.freq = self.freq[self.freq != 0]

    def reset_random_core(self, n=5):
        """
        Reseta a instância com um core aleatório de n átomos como solução inicial.
        """
        core_found = False

        while not core_found:
            self.reset()
            core_found = True
            core_coords = artificial_molecule_coords(n)
            core_all_dist = utils.coords_to_dist(core_coords)

            for core_dist in core_all_dist:
                dist_found = False

                for k in range(self.m):
                    if abs(core_dist - self.dist[k]) < 0.1:
                        self.freq[k] -= 1
                        dist_found = True
                        break

                if not dist_found:
                    core_found = False

        self.coords = core_coords
        self.remove_zero_freq()

    def append_coords(self, new_coords) -> None:
        """
        Adiciona novas coordenadas à solução atual.
        """
        coords = np.concatenate([self.coords, new_coords.round(3)])
        new_all_dist = utils.coords_pair_to_dist(self.coords, new_coords)
        self.coords = coords

    @classmethod
    def from_coords(cls, coords, freq=True):
        """
        Retorna: instância referente às coordenadas fornecidas.
        """
        all_distances = utils.coords_to_dist(coords)

        if freq:
            dist, freq = np.unique(all_distances, return_counts=True)
        else:
            dist = all_distances
            freq = None

        return cls(coords.shape[0], dist, freq, coords)

    @classmethod
    def artificial_molecule(cls, n: int, seed: int = None, freq=True):
        """
        Retorna: instância de molécula artificial com n átomos.
        """
        coords = artificial_molecule_coords(n, seed)
        return cls.from_coords(coords, freq)
