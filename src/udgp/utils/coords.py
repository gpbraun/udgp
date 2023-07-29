"""Gabriel Braun, 2023

Este módulo implementa funções gerais para manipulação de instâncias do problema uDGP.
"""

import networkx as nx
import numpy as np
import py3Dmol
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from sklearn.neighbors import radius_neighbors_graph


def coords_dists(coords: np.ndarray, return_indices=False):
    """
    Parâmetros:
        - coords (numpy.ndarray): matriz de coordenadas.
        - return_indices (bool): retorna os índices referentes às distâncias.

    Retorna:
        - lista completa ordenada de distâncias entre os vértices.
        - lista de índices referentes às distâncias ordenadas.
    """
    coords = np.atleast_2d(coords)
    distances = squareform(pdist(coords, metric="euclidean"))

    i, j = np.triu_indices(coords.shape[0], k=1)

    sorted_args = np.argsort(distances[i, j])
    sorted_dists = distances[i, j][sorted_args].round(2).astype(np.float16)

    if return_indices:
        sorted_indices = np.c_[i[sorted_args], j[sorted_args]].astype(np.int16)

        return sorted_dists, sorted_indices

    return sorted_dists


def coords_new_dists(x_coords: np.ndarray, y_coords: np.ndarray, return_indices=False):
    """
    Parâmetros:
        - y_coords (numpy.ndarray): matriz de coordenadas fixadas.
        - x_coords (numpy.ndarray): matriz de novas coordenadas.
        - return_indices (bool): retorna os índices referentes às distâncias.

    Retorna:
        - lista completa ordenada de distâncias entre os vértices.
        - lista de índices referentes às distâncias ordenadas.
    """
    x_coords = np.atleast_2d(x_coords)
    y_coords = np.atleast_2d(y_coords)
    coords = np.r_[y_coords, x_coords]
    dists = squareform(pdist(coords, metric="euclidean"))

    n_x, n_y, n = x_coords.shape[0], y_coords.shape[0], coords.shape[0]

    grid = np.mgrid[0:n, n_y:n].reshape(2, -1)
    i, j = grid[:, grid[0] < grid[1]]

    sorted_args = np.argsort(dists[i, j])
    sorted_dists = dists[i, j][sorted_args].round(2).astype(np.float16)

    if return_indices:
        sorted_indices = np.c_[i[sorted_args], j[sorted_args]].astype(np.int16)

        return sorted_dists, sorted_indices

    return sorted_dists


def coords_split(coords: np.ndarray, split_size: int):
    """
    Retorna: coordenadas divididas.
    """
    return train_test_split(coords, test_size=split_size)


def coords_xyz_str(coords: np.ndarray, title="uDGP instance"):
    """
    Retorna: string com a representação da instância no formato xyz.
    """
    xyz_coords = [f"C    {p[0]:.4f}    {p[1]:.4f}    {p[2]:.4f}" for p in coords]
    return "\n".join([str(coords.shape[0]), title, *xyz_coords])


def coords_view(coords: np.ndarray, bg_color="#000000", alpha=0.2):
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
