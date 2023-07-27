"""Gabriel Braun, 2023

Este módulo implementa funções gerais para manipulação de instâncias do problema uDGP.
"""

import networkx as nx
import numpy as np
import py3Dmol
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import radius_neighbors_graph


def coords_dist(coords: np.ndarray) -> np.ndarray:
    """
    Retorna: lista ordenada de distâncias entre os vértices.
    """
    if coords.shape[0] < 1:
        return np.array([], dtype=np.float16)

    repeat_dist = pdist(coords, "euclidean")
    return np.sort(repeat_dist.round(2).astype(np.float16))


def coords_pair_dist(coords_1: np.ndarray, coords_2: np.ndarray) -> np.ndarray:
    """
    Retorna: lista ordenada de distâncias entre os vértices.
    """
    if coords_1.shape[0] == 0 or coords_2.shape[0] == 0:
        return np.array([], dtype=np.float16)

    repeat_dist = cdist(coords_1, coords_2, "euclidean").flatten()
    return np.sort(repeat_dist.round(2).astype(np.float16))


def coords_new_dist(coords_fixed: np.ndarray, coords_new: np.ndarray) -> np.ndarray:
    """
    Retorna: lista ordenada de distâncias entre os vértices.
    """
    fixed_new = coords_pair_dist(coords_fixed, coords_new)
    new_new = coords_dist(coords_new)

    return np.sort(np.concatenate((fixed_new, new_new), dtype=np.float16))


def coords_split(coords: np.ndarray, split_size: int):
    """
    Retorna: coordenadas divididas.
    """
    return train_test_split(coords, test_size=split_size)


def coords_xyz_str(coords: np.ndarray, title="uDGP instance") -> str:
    """
    Retorna: string com a representação da instância no formato xyz.
    """
    xyz_coords = [f"C    {p[0]:.4f}    {p[1]:.4f}    {p[2]:.4f}" for p in coords]
    return "\n".join([str(coords.shape[0]), title, *xyz_coords])


def coords_view(coords: np.ndarray, bg_color="#000000", alpha=0.2) -> py3Dmol.view:
    """
    Retorna: visualização da instância com py3Dmol.
    """
    xyz_str = coords_xyz_str(coords)
    view = py3Dmol.view(data=xyz_str, width=400, height=350)
    view.setBackgroundColor(bg_color, alpha)
    view.setStyle(
        {
            "stick": {"radius": 0.1, "color": "#cbd5e1"},
            "sphere": {"scale": 0.2, "color": "#60a5fa"},
        }
    )
    return view


# def coords_adjacency_matrix(coords: np.ndarray) -> csr_matrix:
#     """
#     Retorna: matriz de adjacência da instância.
#     """
#     return radius_neighbors_graph(coords, 1.8, mode="connectivity")

# def coords_graph(coords: np.ndarray) -> nx.Graph:
#     """
#     Retorna: representação de grafo da instância.
#     """
#     am = coords_adjacency_matrix(coords)
#     return nx.from_scipy_sparse_array(am)

# def coords_are_isomorphic(coords_1, coords_2) -> bool:
#     """
#     Retorna: verdadeiro se as coordenadas representam a mesma molécula.
#     """
#     graph_1 = coords_graph(coords_1)
#     graph_2 = coords_graph(coords_2)
#     return nx.vf2pp_is_isomorphic(graph_1, graph_2, node_label=None)
