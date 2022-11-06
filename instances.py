"""Criação de instâncias."""

import numpy as np

from itertools import combinations


np.random.seed(12345)


class Atom:
    """Um átomo."""

    def __init__(self, coord):
        self.x = coord[0]
        self.y = coord[1]
        self.z = coord[2]

    def distance(self, other):
        delta_x = self.x - other.x
        delta_y = self.y - other.y
        delta_z = self.z - other.z

        return round(np.sqrt(delta_x**2 + delta_y**2 + delta_z**2), 4)


def B(n: int, r: float, theta: float, omega: float) -> np.ndarray:
    """Retorna a matriz B."""

    if n == 0:
        return np.identity(4)

    if n == 1:
        return np.array([[-1, 0, 0, -r], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    if n == 2:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return np.array(
            [
                [-cos_theta, -sin_theta, 0, -r * cos_theta],
                [sin_theta, -cos_theta, 0, -r * sin_theta],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

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


def B_prod(
    i: int, lengths: np.ndarray, thetas: np.ndarray, omegas: np.ndarray
) -> np.ndarray:
    """Retorna o produto das matrizes B de 0 a i."""
    if i == 0:
        return B(0, lengths[0], thetas[0], omegas[0])

    return B_prod(i - 1, lengths, thetas, omegas) @ B(
        i, lengths[i], thetas[i], omegas[i]
    )


def create_instance(lengths: np.ndarray, thetas: np.ndarray, omegas: np.ndarray):
    """Cria uma instância a partir das disânticas, ângulos de ligação e ângulos de torção."""
    n = len(lengths)

    points = [
        B_prod(i, lengths, thetas, omegas).dot(np.array([0, 0, 0, 1]).T)[0:3]
        for i in range(n)
    ]

    xyz_coords = [f"C    {p[0]:.4f}    {p[1]:.4f}    {p[2]:.4f}" for p in points]
    xyz_str = "\n".join([str(n), "INSTANCE", *xyz_coords])

    ## Salvar coordenadas
    with open("input.xyz", "w") as f:
        f.write(xyz_str)

    atoms = map(Atom, points)

    d_str = "\n".join([str(x.distance(y)) for x, y in combinations(atoms, 2)])

    ## Salvar distâncias
    with open("distances.txt", "w") as f:
        f.write(d_str)

    return atoms


OMEGA_VALUES = np.array([np.pi / 3.0, np.pi, 5.0 * np.pi / 3.0])


def main():
    n = 5

    lengths = np.full(n, 1.5)
    thetas = np.full(n, 2)
    omegas = np.random.choice(OMEGA_VALUES, size=n)

    create_instance(lengths, thetas, omegas)


if __name__ == "__main__":
    main()
