from importlib import resources

import numpy as np

_DATA_PATH = resources.files("udgp.instances.data")
"""
Diretório da base de dados.
"""


def c60_points():
    """
    Retorna: matriz de coordenadas do C60.
    """
    points = np.loadtxt(_DATA_PATH.joinpath(f"c60/c60.txt"), dtype=np.float64)

    return points
