import importlib.resources

import numpy as np

LJ_CLUSTER_PATH = importlib.resources.files("udgp.data.lj_cluster")
"""Diretório da base de dados."""


def lj_cluster_points(n: int):
    """
    Retorna: matriz de coordenadas de um cluster de Lennard-Jones com n (entre 3 e 150) átomos.

    Referência: https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html
    """
    points = np.loadtxt(LJ_CLUSTER_PATH.joinpath(f"lj_{n}.txt"))

    return points
