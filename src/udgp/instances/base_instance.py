"""Gabriel Braun, 2023

Este módulo implementa a classe base para instâncias do problema uDGP.
"""

import networkx as nx
import numpy as np
import py3Dmol
from scipy.sparse import csr_matrix
from sklearn.neighbors import radius_neighbors_graph


class Instance:
    """Instância para o problema uDGP."""

    def __init__(self, atoms: np.ndarray, distances: np.ndarray):
        self.atoms = atoms
        self.distances = distances
        self.n = self.atoms.shape[0]
        self.m = self.distances.shape[0]

    def is_isomorphic(self, other) -> bool:
        """Retorna: verdadeiro se as instâncias representam a mesma molécula."""
        return nx.vf2pp_is_isomorphic(self.graph(), other.graph(), node_label=None)

    def graph(self) -> nx.Graph:
        """Retorna: representação de grafo da instância."""
        return nx.from_scipy_sparse_array(self.adjacency_matrix())

    def adjacency_matrix(self) -> csr_matrix:
        """Retorna: matriz de adjacência da instância"""
        return radius_neighbors_graph(self.atoms, 1.8, mode="connectivity")

    def xyz_str(self, title="Instância para uDGP") -> str:
        """Retorna: string com a representação da instância no formato xyz."""
        xyz_coords = [
            f"C    {p[0]:.4f}    {p[1]:.4f}    {p[2]:.4f}" for p in self.atoms
        ]
        return "\n".join([str(self.n), title, *xyz_coords])

    def view(self, bg_color="#000000") -> py3Dmol.view:
        """Retorna: visualização da instância com py3Dmol."""
        input_xyz_str = self.xyz_str()
        view = py3Dmol.view(data=input_xyz_str)
        view.setBackgroundColor(bg_color)
        view.setStyle(
            {
                "stick": {"radius": 0.1},
                "sphere": {"scale": 0.2},
            }
        )
        return view
