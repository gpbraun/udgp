"""Gabriel Braun, 2023

Este módulo implementa a classe base para instâncias do problema uDGP.
"""

from itertools import chain, combinations, product

import networkx as nx
import numpy as np
import py3Dmol
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import radius_neighbors_graph


def split_coords(coords: np.ndarray, split_size: int):
    """Retorna: coordenadas divididas."""
    return train_test_split(coords, test_size=split_size)


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
    view = py3Dmol.view(data=xyz_str, width=300, height=300)
    view.setBackgroundColor(bg_color, alpha)
    view.setStyle(
        {
            "stick": {"radius": 0.1},
            "sphere": {"scale": 0.2},
        }
    )
    return view


def calculate_norm(vectors: np.ndarray):
    """Retorna a lista de normas a partir de uma lista de vetores.

    Ref. https://stackoverflow.com/questions/14758283
    """
    return np.sort(np.sqrt(np.einsum("ij,ij->i", vectors, vectors))).astype("float16")


class Instance:
    """Instância para o problema uDGP."""

    def __init__(
        self,
        n: int,
        distances: np.ndarray = np.empty(0),
        coords: np.ndarray = np.zeros((1, 3)),
        input_coords: np.ndarray | None = None,
    ):
        self.n = n
        self.m = distances.shape[0]
        self.distances = distances
        self.coords = coords
        self.input_coords = input_coords

    def view(self) -> py3Dmol.view:
        """Retorna: visualização da instância com py3Dmol."""
        return coords_to_view(self.coords)

    def view_input(self) -> py3Dmol.view:
        """Retorna: visualização da instância com py3Dmol."""
        if self.input_coords is None:
            return

        return coords_to_view(self.input_coords)

    def is_isomorphic(self) -> bool:
        """Retorna: verdadeiro as coordenadas representam a mesma molécula que o input."""
        if self.input_coords is None:
            return False

        return coords_are_isomorphic(self.coords, self.input_coords)

    def mock_core(self, core_size: int = 5):
        """Transforma a instância em um núcleo para testes de heurísticas."""
        if self.input_coords is None:
            return

        self.coords = split_coords(self.input_coords, core_size)[1]
        # ISSO AQUI FOI FEITO PELA BIA MALUCA:
        # As distâncias devem ser calculadas como foi feito no colab!!!
        self.distances = np.sort(pdist(self.coords, "euclidean"))
        self.m = self.distances.shape[0]
