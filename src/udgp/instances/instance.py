"""Gabriel Braun, 2023

Este módulo implementa a classe base para instâncias do problema uDGP.
"""

import networkx as nx
import numpy as np
import py3Dmol
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import radius_neighbors_graph

from .random import random_coords

START_COORDS = np.array([[0.0, 0.0, 0.0]])


def split_coords(coords: np.ndarray, split_size: int):
    """Retorna: coordenadas divididas."""
    return train_test_split(coords, test_size=split_size)


def coords_to_dist(coords: np.ndarray):
    """Retorna: lista ordenada de distâncias entre os vértices."""
    return np.sort(pdist(coords, "euclidean").round(3).astype(np.float16))


def coords_to_adjacency_matrix(coords: np.ndarray) -> csr_matrix:
    """Retorna: matriz de adjacência da instância."""
    return radius_neighbors_graph(coords, 1.8, mode="connectivity")


def coords_to_graph(coords: np.ndarray) -> nx.Graph:
    """Retorna: representação de grafo da instância."""
    am = coords_to_adjacency_matrix(coords)
    return nx.from_scipy_sparse_array(am)


def coords_are_isomorphic(coords_1, coords_2) -> bool:
    """Retorna: verdadeiro se as coordenadas representam a mesma molécula."""
    graph_1 = coords_to_graph(coords_1)
    graph_2 = coords_to_graph(coords_2)
    return nx.vf2pp_is_isomorphic(graph_1, graph_2, node_label=None)


def coords_to_xyz_str(coords, title="uDGP instance") -> str:
    """Retorna: string com a representação da instância no formato xyz."""
    xyz_coords = [f"C    {p[0]:.4f}    {p[1]:.4f}    {p[2]:.4f}" for p in coords]
    return "\n".join([str(coords.shape[0]), title, *xyz_coords])


def coords_to_view(coords, bg_color="#000000", alpha=0.2) -> py3Dmol.view:
    """Retorna: visualização da instância com py3Dmol."""
    xyz_str = coords_to_xyz_str(coords)
    view = py3Dmol.view(data=xyz_str, width=400, height=350)
    view.setBackgroundColor(bg_color, alpha)
    view.setStyle(
        {
            "stick": {"radius": 0.1, "color": "#cbd5e1"},
            "sphere": {"scale": 0.2, "color": "#60a5fa"},
        }
    )
    return view


class Instance:
    """Instância para o problema uDGP."""

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
        """Retorna: número de distâncias."""
        return self.dist.shape[0]

    @property
    def fixed_n(self) -> int:
        """Retorna: número de átomos fixados."""
        return self.coords.shape[0]

    @property
    def all_dist(self) -> np.ndarray:
        """Retorna: lista completa (ordenada) de distâncias remanescentes."""
        return np.repeat(self.dist, self.freq)

    @property
    def input_all_dist(self) -> np.ndarray:
        """Retorna: lista completa (ordenada) de distâncias de entrada."""
        return np.repeat(self.input_dist, self.input_freq)

    def view(self) -> py3Dmol.view:
        """Retorna: visualização da instância com py3Dmol."""
        return coords_to_view(self.coords)

    def view_input(self) -> py3Dmol.view:
        """Retorna: visualização da instância com py3Dmol."""
        if self.input_coords is None:
            return

        return coords_to_view(self.input_coords)

    def is_solved(self, threshold=1e-3) -> bool:
        """Retorna: verdadeiro se as distâncias do input são as mesmas da da solução.

        Referência: LIGA
        """
        solution_all_dist = coords_to_dist(self.coords)

        if solution_all_dist.shape != self.input_all_dist.shape:
            return False

        var = np.var(solution_all_dist - self.input_all_dist)

        return var < threshold

    def is_isomorphic(self) -> bool:
        """Retorna: verdadeiro as coordenadas representam a mesma molécula que o input."""
        if self.input_coords is None:
            return False

        return coords_are_isomorphic(self.coords, self.input_coords)

    def get_random_coords(self, n=4) -> np.ndarray:
        """Retorna: n coordenadas já fixadas escolhidas aleatoriamente."""
        if n >= self.coords.shape[0]:
            return self.coords

        sample_coords = split_coords(self.coords, n)[1]
        return sample_coords

    def reset(self) -> None:
        """Reseta a instância para o estado inicial."""
        self.dist = self.input_dist.copy()
        self.freq = self.input_freq.copy()
        self.coords = START_COORDS.copy()

    def reset_random_core(self, n=5):
        """Reseta a instância com um core aleatório de n átomos."""
        core_found = False

        while not core_found:
            self.reset()
            core_found = True
            core_coords = random_coords(n)
            core_all_dist = coords_to_dist(core_coords)

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
        coords = np.concatenate([self.coords, new_coords])
        self.coords = np.unique(coords.round(3), axis=0)

    def remove_zero_freq(self):
        self.dist = self.dist[self.freq != 0]
        self.freq = self.freq[self.freq != 0]

    @classmethod
    def from_coords(cls, coords, freq=True):
        """Retorna: instância referente às coordenadas fornecidas."""
        all_distances = coords_to_dist(coords)

        if freq:
            dist, freq = np.unique(all_distances, return_counts=True)
        else:
            dist = all_distances
            freq = None

        return cls(coords.shape[0], dist, freq, coords)

    @classmethod
    def random(cls, n: int, seed: int = None, freq=True):
        """Retorna: instância aleatória com n átomos."""
        coords = random_coords(n, seed)
        return cls.from_coords(coords, freq)
