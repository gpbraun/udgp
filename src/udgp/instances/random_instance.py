"""Gabriel Braun, 2023

Este módulo implementa funções para geração de instâncias aleatórias do uDGP.

Referência: Lavor, C. (2006) https://doi.org/10.1007/0-387-30528-9_14
"""

import numpy as np
from scipy.spatial.distance import pdist

from .base_instance import Instance

BOND_LENGTH_VALUES = np.array([1.5])
BOND_ANGLE_VALUES = np.array([np.arccos(-1 / 3)])
TORSION_ANGLE_VALUES = np.array([np.pi / 3, np.pi, 5 * np.pi / 3])


def b_matrix(
    i: int,
    bond_lengths: np.ndarray,
    bond_angles: np.ndarray,
    torsion_angles: np.ndarray,
) -> np.ndarray:
    """Retorna: matriz B de índice i."""
    if i == 0:
        return np.identity(4)

    r = bond_lengths[i - 1]

    if i == 1:
        return np.array(
            [
                [-1, 0, 0, -r],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )

    cos_theta = np.cos(bond_angles[i - 2])
    sin_theta = np.sin(bond_angles[i - 2])

    if i == 2:
        return np.array(
            [
                [-cos_theta, -sin_theta, 0, -r * cos_theta],
                [sin_theta, -cos_theta, 0, r * sin_theta],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    cos_omega = np.cos(torsion_angles[i - 3])
    sin_omega = np.sin(torsion_angles[i - 3])

    return np.array(
        [
            [-cos_theta, -sin_theta, 0, -r * cos_theta],
            [
                sin_theta * cos_omega,
                -cos_theta * cos_omega,
                -sin_omega,
                r * sin_theta * cos_omega,
            ],
            [
                sin_theta * sin_omega,
                -cos_theta * sin_omega,
                cos_omega,
                r * sin_theta * sin_omega,
            ],
            [0, 0, 0, 1],
        ]
    )


def b_matrix_product(
    i: int,
    bond_lengths: np.ndarray,
    bond_angles: np.ndarray,
    torsion_angles: np.ndarray,
) -> np.ndarray:
    """Retorna: produto das matrizes B de 0 a i."""
    if i == 0:
        return b_matrix(0, bond_lengths, bond_angles, torsion_angles)

    product = b_matrix_product(i - 1, bond_lengths, bond_angles, torsion_angles)

    return product @ b_matrix(i, bond_lengths, bond_angles, torsion_angles)


def random_coords(n: int, seed: int = None) -> np.ndarray:
    """Define as coordenadas dos átomos a partir dos comprimentos de ligação, ângulos de ligação e ângulos de torção.

    Retorna: matriz de coordenadas dos átomos.
    """
    rng = np.random.default_rng(seed)

    bond_lengths = rng.choice(BOND_LENGTH_VALUES, size=(2 * n))
    bond_angles = rng.choice(BOND_ANGLE_VALUES, size=(2 * n))
    torsion_angles = rng.choice(TORSION_ANGLE_VALUES, size=(2 * n))

    coords = np.array(
        [
            (
                b_matrix_product(i, bond_lengths, bond_angles, torsion_angles)
                @ np.array([0, 0, 0, 1]).T
            )[0:3]
            for i in range(n)
        ]
    )

    return np.unique(coords.round(decimals=4), axis=0)


def random_instance(n: int, seed: int = None, freq=True) -> Instance:
    """Cria uma instância aleatória com n átomos."""
    coords = random_coords(n, seed)

    return Instance.from_coords(coords, freq)
