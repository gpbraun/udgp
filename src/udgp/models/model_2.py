"""Gabriel Braun, 2023

Este módulo implementa o modelo M4 para instâncias do problema uDGP.
"""

from .model_1 import M1


class M2(M1):
    """Modelo M2 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(M2, self).__init__(*args, **kwargs)
        self.relax()
        self.update()
