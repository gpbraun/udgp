"""Gabriel Braun, 2023

Este módulo implementa a classe base para instâncias do problema uDGP.
"""

import numpy as np

from udgp.utils import *

from .artificial_molecule import artificial_molecule_coords


class Instance:
    """
    Instância para o problema uDGP.
    """

    def __init__(
        self,
        n: int,
        dists: np.ndarray,
        freqs: np.ndarray | None = None,
        coords: np.ndarray | None = None,
    ):
        if freqs is None:
            freqs = np.ones_like(dists, dtype=np.int64)

        self.n = n

        self.dists = dists
        self.freqs = freqs
        self.coords = np.zeros((1, 3), dtype=np.float16)

        self.a_indices = np.empty((0, 3), dtype=np.int16)

        self.input_dists = dists
        self.input_freqs = freqs
        self.input_coords = coords

    @property
    def m(self):
        """
        Retorna: número de distsâncias remanescentes.
        """
        return self.dists.shape[0]

    @property
    def fixed_n(self):
        """
        Retorna: número de átomos da solução atual (fixados).
        """
        return self.coords.shape[0]

    @property
    def repeat_dists(self):
        """
        Retorna: lista com repetição (ordenada) de distsâncias remanescentes.
        """
        return np.repeat(self.dists, self.freqs)

    @property
    def input_repeat_dists(self):
        """
        Retorna: lista com repetição (ordenada) de distsâncias de entrada.
        """
        return np.repeat(self.input_dists, self.input_freqs)

    def view(self):
        """
        Retorna: visualização da solução com py3Dmol.
        """
        return coords_view(self.coords)

    def view_input(self):
        """
        Retorna: visualização da instância com py3Dmol.
        """
        if self.input_coords is None:
            return

        return coords_view(self.input_coords)

    def is_solved(self, threshold=1e-3):
        """
        Retorna: verdadeiro se as distsâncias do input são as mesmas da da solução.

        Referência: LIGA
        """
        solution_repeat_dists = coords_dists(self.coords)

        if solution_repeat_dists.shape != self.input_repeat_dists.shape:
            return False

        var = np.var(solution_repeat_dists - self.input_repeat_dists)

        return var < threshold

    def reset(self):
        """
        Reseta a instância para o estado inicial.
        """
        self.dists = self.input_dists.copy()
        self.freqs = self.input_freqs.copy()
        self.coords = np.zeros((1, 3), dtype=np.float16)
        self.a_indices = np.empty((0, 3), dtype=np.int16)

    def remove_coords_dists(self, dists: np.ndarray, indices: np.ndarray):
        """
        Remove uma lista distâncias com repetições da lista de distsâncias remanescentes.

        Parâmetros:
        - dists (numpy.ndarray): lista de distância com repetições para serem removidas
        - indices (numpy.ndarray): índices referentes às distâncias.

        Retorna: verdadeiro se todas as distsâncias fornecidas estavam na lista de distsâncias remanescentes.
        """
        new_freqs = self.freqs.copy()

        for dist, (i, j) in zip(dists, indices):
            if dist == 0:
                return False

            errors = np.abs(1 - self.dists / dist)
            k = errors.argmin()

            if errors[k] < 0.2 and new_freqs[k] > 0:
                new_freqs[k] -= 1
                self.a_indices = np.vstack((self.a_indices, [i, j, k]))
            else:
                print(f"distância {dist} não está na lista: {self.dists}")
                print(f"frequências: {new_freqs}")
                print(f"erro mínimo: {self.dists[k]} -> {errors[k]}")
                return False

        self.freqs = new_freqs
        return True

    def add_coords(self, new_coords: np.ndarray) -> bool:
        """
        Adiciona (fixa) novas coordenadas à solução.

        Retorna: verdadeiro se as distsâncias entre as novas coordenadas e as coordenadas já fixadas estavam na lista de distsâncias remanescentes.
        """
        new_dists, new_indices = coords_new_dists(
            new_coords, self.coords, return_indices=True
        )
        print(f"novas distâncias: {new_dists}")
        print(f"distâncias disponíveis: {self.input_repeat_dists}")

        if self.remove_coords_dists(new_dists, new_indices):
            self.coords = np.r_[self.coords, new_coords.round(3)]
            return True

        print(f"coordenadas:\n{new_coords}\nnão puderam ser adicionadas.")
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

            core_dists, core_indices = coords_dists(core_coords, return_indices=True)
            core_found = self.remove_coords_dists(core_dists, core_indices)

        self.coords = core_coords

    @classmethod
    def from_coords(cls, coords, freq=True):
        """
        Retorna: instância referente às coordenadas fornecidas.
        """
        dists = coords_dists(coords)
        freqs = None

        if freq:
            dists, freqs = np.unique(dists, return_counts=True)

        return cls(coords.shape[0], dists, freqs, coords)

    @classmethod
    def artificial_molecule(cls, n: int, seed: int = None, freq=True):
        """
        Retorna: instância de molécula artificial com n átomos.

        Referência: Lavor, C. (2006) https://doi.org/10.1007/0-387-30528-9_14
        """
        coords = artificial_molecule_coords(n, seed)
        return cls.from_coords(coords, freq)
