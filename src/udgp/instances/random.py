"""Gabriel Braun, 2023

Este módulo implementa funções para geração de instâncias aleatórias do uDGP.

Referência: Lavor, C. (2006) https://doi.org/10.1007/0-387-30528-9_14
"""

import numpy as np

BOND_LENGTH_VALUES = np.array([1.5])
BOND_ANGLE_VALUES = np.array([np.arccos(-1 / 3)])
TORSION_ANGLE_VALUES = np.array([np.pi / 3, np.pi, 5 * np.pi / 3])


def b_matrix(i: int, seed: int = None) -> np.ndarray:
    """Retorna: matriz B de índice i."""
    if i == 0:
        return np.identity(4)

    rng = np.random.default_rng(seed)
    r = rng.choice(BOND_LENGTH_VALUES)

    if i == 1:
        return np.array(
            [
                [-1, 0, 0, -r],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )

    theta = rng.choice(BOND_ANGLE_VALUES)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    if i == 2:
        return np.array(
            [
                [-cos_theta, -sin_theta, 0, -r * cos_theta],
                [sin_theta, -cos_theta, 0, r * sin_theta],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    omega = rng.choice(TORSION_ANGLE_VALUES)
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)

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


def random_coords(n: int, seed: int = None) -> np.ndarray:
    """Retorna: matriz de coordenadas criada aletatóriamente."""
    coords = np.empty((n, 3), dtype=np.float16)

    b = b_matrix(0, seed)
    col = np.array([0, 0, 0, 1]).T

    for i in range(n):
        np.matmul(b, b_matrix(i, seed), out=b)
        coords[i] = (b @ col)[0:3].round(3)

    return coords
