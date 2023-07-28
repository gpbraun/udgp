"""Gabriel Braun, 2023

Este módulo implementa a classe base para instâncias do problema uDGP.
"""

import numpy as np
from py3Dmol import view

from udgp.utils import *

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
        self.a_ijk = []
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
    def repeat_dist(self) -> np.ndarray:
        """
        Retorna: lista com repetição (ordenada) de distâncias remanescentes.
        """
        return np.repeat(self.dist, self.freq)

    @property
    def input_repeat_dist(self) -> np.ndarray:
        """
        Retorna: lista com repetição (ordenada) de distâncias de entrada.
        """
        return np.repeat(self.input_dist, self.input_freq)

    def view(self) -> view:
        """
        Retorna: visualização da solução com py3Dmol.
        """
        return coords_view(self.coords)

    def view_input(self) -> view:
        """
        Retorna: visualização da instância com py3Dmol.
        """
        if self.input_coords is None:
            return

        return coords_view(self.input_coords)

    def is_solved(self, threshold=1e-3) -> bool:
        """
        Retorna: verdadeiro se as distâncias do input são as mesmas da da solução.

        Referência: LIGA
        """
        solution_repeat_dist = coords_dist(self.coords)

        if solution_repeat_dist.shape != self.input_repeat_dist.shape:
            return False

        var = np.var(solution_repeat_dist - self.input_repeat_dist)

        return var < threshold

    def reset(self):
        """
        Reseta a instância para o estado inicial.
        """
        self.dist = self.input_dist.copy()
        self.freq = self.input_freq.copy()
        self.coords = START_COORDS.copy()
        self.a_ijk = []

    def remove_distances(self, new_repeat_dist: np.ndarray) -> bool:
        """
        Remove uma lista distâncias com repetições da lista de distâncias remanescentes.

        Retorna: verdadeiro se todas as distâncias fornecidas estavam na lista de distâncias remanescentes.
        """
        new_freq = self.freq.copy()

        for new_dist in new_repeat_dist:
            if new_dist == 0:
                return False

            errors = np.abs(1 - self.dist / new_dist)
            k = errors.argmin()

            if errors[k] < 0.2 and new_freq[k] > 0:
                new_freq[k] -= 1
            else:
                print(f"distância {new_dist} não está na lista: {self.dist}")
                print(f"frequências: {new_freq}")
                print(f"erro mínimo: {self.dist[k]} -> {errors[k]}")
                return False

        self.freq = new_freq
        return True

    def add_coords(self, new_coords: np.ndarray) -> bool:
        """
        Adiciona (fixa) novas coordenadas à solução.

        Retorna: verdadeiro se as distâncias entre as novas coordenadas e as coordenadas já fixadas estavam na lista de distâncias remanescentes.
        """
        new_repeat_dist = coords_new_dist(self.coords, new_coords)

        if self.remove_distances(new_repeat_dist):
            self.coords = np.concatenate(
                (self.coords, new_coords.round(3)), dtype=np.float16
            )
            return True

        print(f"coordenadas {new_coords} não puderam ser adicionadas.")
        return False

    def reset_with_core(self, core_type: str, n=5):
        """
        Reseta a instância com um core de molécula artificial de n átomos como solução inicial.
        """
        core_found = False

        while not core_found:
            self.reset()

            if core_type == "mock":
                core_coords = coords_split(self.input_coords, n)[1]
            if core_type == "artificial":
                core_coords = artificial_molecule_coords(n)

            core_repeat_dist = coords_dist(core_coords)
            core_found = self.remove_distances(core_repeat_dist)

        self.coords = core_coords

    @classmethod
    def from_coords(cls, coords, freq=True):
        """
        Retorna: instância referente às coordenadas fornecidas.
        """
        repeat_distances = coords_dist(coords)

        if freq:
            dist, freq = np.unique(repeat_distances, return_counts=True)
        else:
            dist = repeat_distances
            freq = None

        return cls(coords.shape[0], dist, freq, coords)

    @classmethod
    def artificial_molecule(cls, n: int, seed: int = None, freq=True):
        """
        Retorna: instância de molécula artificial com n átomos.

        Referência: Lavor, C. (2006) https://doi.org/10.1007/0-387-30528-9_14
        """
        coords = artificial_molecule_coords(n, seed)
        return cls.from_coords(coords, freq)
