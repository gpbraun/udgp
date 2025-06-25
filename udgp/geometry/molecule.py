"""
Gabriel Braun, 2025
"""

__all__ = [
    "geom_molecule_c60",
]

from importlib import resources

import numpy as np

_DATA_PATH = resources.files("udgp.geometry.molecule")
"""
Diretório da base de dados.
"""


def geom_molecule_c60():
    """
    Retorna: matriz de coordenadas do C60.
    """
    points = np.loadtxt(_DATA_PATH.joinpath(f"c60.txt"), dtype=np.float64)

    return points
