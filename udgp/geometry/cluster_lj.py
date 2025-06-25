from importlib import resources

import numpy as np

_DATA_PATH = resources.files("udgp.geometry.data.lj_cluster")
"""
Diretório da base de dados.
"""


def geom_cluster_lj(n: int):
    """
    Retorna: matriz de coordenadas de um cluster de Lennard-Jones com n (entre 3 e 150) átomos.

    Referência: https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html
    """
    points = np.loadtxt(_DATA_PATH.joinpath(f"lj_{n}.txt"), dtype=np.float64)

    return points
