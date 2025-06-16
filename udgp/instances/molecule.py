import importlib.resources

import numpy as np

DATA_PATH = importlib.resources.files("udgp.instances.data")
"""Diretório da base de dados."""


def c60_points():
    """
    Retorna: matriz de coordenadas do C60.
    """
    points = np.loadtxt(DATA_PATH.joinpath(f"c60/c60.txt"), dtype=np.float64)

    return points
