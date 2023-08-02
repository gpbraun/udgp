import importlib.resources

import numpy as np

DATA_PATH = importlib.resources.files("udgp.data")
"""Diretório da base de dados."""


def lj_cluster_points(n: int):
    """
    Retorna: matriz de coordenadas de um cluster de Lennard-Jones com n (entre 3 e 150) átomos.

    Referência: https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html
    """
    points = np.loadtxt(DATA_PATH.joinpath(f"lj_cluster/lj_{n}.txt"))

    return points


def c60():
    """
    Retorna: matriz de coordenadas do C60.
    """
    points = np.loadtxt(DATA_PATH.joinpath(f"c60/c60.txt"))

    return points
