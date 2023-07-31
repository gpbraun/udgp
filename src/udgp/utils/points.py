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


def points_dists(points: np.ndarray, return_indices=False):
    """
    Parâmetros:
        - points (numpy.ndarray): matriz de pointenadas.
        - return_indices (bool): retorna os índices referentes às distâncias.

    Retorna:
        - lista completa ordenada de distâncias entre os vértices.
        - lista de índices referentes às distâncias ordenadas.
    """
    points = np.atleast_2d(points)
    distances = squareform(pdist(points, metric="euclidean"))

    i, j = np.triu_indices(points.shape[0], k=1)

    sorted_args = np.argsort(distances[i, j])
    sorted_dists = distances[i, j][sorted_args].round(2).astype(np.float16)

    if return_indices:
        sorted_indices = np.c_[i[sorted_args], j[sorted_args]].astype(np.int16)

        return sorted_dists, sorted_indices

    return sorted_dists


def points_new_dists(
    x_points: np.ndarray, y_points: np.ndarray | None = None, return_indices=False
):
    """
    Parâmetros:
        - y_points (numpy.ndarray): matriz de pointenadas fixadas.
        - x_points (numpy.ndarray): matriz de novas pointenadas.
        - return_indices (bool): retorna os índices referentes às distâncias.

    Retorna:
        - lista completa ordenada de distâncias entre os vértices.
        - lista de índices referentes às distâncias ordenadas.
    """
    x_points = np.atleast_2d(x_points)
    y_points = np.atleast_2d(y_points)
    points = np.r_[y_points, x_points]
    dists = squareform(pdist(points, metric="euclidean"))

    n_y, n = y_points.shape[0], points.shape[0]

    grid = np.mgrid[0:n, n_y:n].reshape(2, -1)
    i, j = grid[:, grid[0] < grid[1]]

    sorted_args = np.argsort(dists[i, j])
    sorted_dists = dists[i, j][sorted_args].round(2).astype(np.float16)

    if return_indices:
        sorted_indices = np.c_[i[sorted_args], j[sorted_args]].astype(np.int16)

        return sorted_dists, sorted_indices

    return sorted_dists


def points_split(points: np.ndarray, split_size: int):
    """
    Retorna: pointenadas divididas.
    """
    return train_test_split(points, test_size=split_size)


def points_xyz_str(points: np.ndarray, title="uDGP instance"):
    """
    Retorna: string com a representação da instância no formato xyz.
    """
    xyz_points = [f"C    {p[0]:.4f}    {p[1]:.4f}    {p[2]:.4f}" for p in points]
    return "\n".join([str(points.shape[0]), title, *xyz_points])


def points_view(points: np.ndarray, bg_color="#000000", alpha=0.2):
    """
    Retorna: visualização da instância com py3Dmol.
    """
    xyz_str = points_xyz_str(points)
    view = py3Dmol.view(data=xyz_str, width=400, height=350)
    view.setBackgroundColor(bg_color, alpha)
    view.setStyle(
        {
            "stick": {"radius": 0.1, "color": "#cbd5e1"},
            "sphere": {"scale": 0.2, "color": "#60a5fa"},
        }
    )
    return view


# def points_adjacency_matrix(points: np.ndarray) -> csr_matrix:
#     """
#     Retorna: matriz de adjacência da instância.
#     """
#     return radius_neighbors_graph(points, 1.8, mode="connectivity")

# def points_graph(points: np.ndarray) -> nx.Graph:
#     """
#     Retorna: representação de grafo da instância.
#     """
#     am = points_adjacency_matrix(points)
#     return nx.from_scipy_sparse_array(am)

# def points_are_isomorphic(points_1, points_2) -> bool:
#     """
#     Retorna: verdadeiro se as pointenadas representam a mesma molécula.
#     """
#     graph_1 = points_graph(points_1)
#     graph_2 = points_graph(points_2)
#     return nx.vf2pp_is_isomorphic(graph_1, graph_2, node_label=None)
