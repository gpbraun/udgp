"""Gabriel Braun, 2023

Este módulo implementa a classe base para instâncias do problema uDGP.
"""

from itertools import combinations

import numpy as np
import py3Dmol


class Instance:
    """Instância para o problema uDGP."""

    def __init__(self, atoms: np.ndarray, distances: np.ndarray):
        self.atoms = atoms
        self.distances = distances
        self.n = self.atoms.shape[0]
        self.m = self.distances.shape[0]

    def to_xyz_str(self, title="Instância para uDGP"):
        """Exporta uma string da instância no formato xyz."""
        xyz_coords = [
            f"C    {p[0]:.4f}    {p[1]:.4f}    {p[2]:.4f}" for p in self.atoms
        ]
        return "\n".join([str(self.n), title, *xyz_coords])

    def view(self, bg_color="#000000"):
        """Visualização da instância com py3Dmol."""
        input_xyz_str = self.to_xyz_str()
        view = py3Dmol.view(data=input_xyz_str)
        view.setBackgroundColor(bg_color)
        view.setStyle(
            {
                "stick": {"radius": 0.1},
                "sphere": {"scale": 0.2},
            }
        )
        return view
