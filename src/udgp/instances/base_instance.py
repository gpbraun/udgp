"""Gabriel Braun, 2023

Este módulo implementa a classe base para instâncias do problema uDGP.
"""

import networkx as nx
import numpy as np
import py3Dmol
from scipy.sparse import csr_matrix
from sklearn.neighbors import radius_neighbors_graph


def coords_to_adjacency_matrix(coords) -> csr_matrix:
    """Retorna: matriz de adjacência da instância"""
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


def coords_to_xyz_str(coords, title=" ") -> str:
    """Retorna: string com a representação da instância no formato xyz."""
    xyz_coords = [f"C    {p[0]:.4f}    {p[1]:.4f}    {p[2]:.4f}" for p in coords]
    return "\n".join([str(coords.shape[0]), title, *xyz_coords])


def coords_to_view(coords, bg_color="#000000") -> py3Dmol.view:
    """Retorna: visualização da instância com py3Dmol."""
    xyz_str = coords_to_xyz_str(coords)
    view = py3Dmol.view(data=xyz_str)
    view.setBackgroundColor(bg_color)
    view.setStyle(
        {
            "stick": {"radius": 0.1},
            "sphere": {"scale": 0.2},
        }
    )
    return view


class Instance:
    """Instância para o problema uDGP."""

    def __init__(
        self,
        n: int,
        distances: np.ndarray = np.empty(0),
        coords: np.ndarray | None = None,
        input_coords: np.ndarray | None = None,
    ):
        self.n = n
        self.m = distances.shape[0]
        self.distances = distances
        self.coords = coords
        self.input_coords = input_coords

    def view_input(self) -> py3Dmol.view:
        """Retorna: visualização da instância com py3Dmol."""
        if self.input_coords is None:
            return

        return coords_to_view(self.input_coords)

    def view(self) -> py3Dmol.view:
        """Retorna: visualização da instância com py3Dmol."""
        if self.coords is None:
            return self.view_input()

        return coords_to_view(self.coords)

    def is_isomorphic(self) -> bool:
        """Retorna: verdadeiro as coordenadas representam a mesma molécula que o input."""
        if self.coords is None or self.input_coords is None:
            return False

        return coords_are_isomorphic(self.coords, self.input_coords)
