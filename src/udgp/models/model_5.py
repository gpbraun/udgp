"""Gabriel Braun, 2023

Este módulo implementa o modelo M4 para instâncias do problema uDGP.
"""

from .model_4 import M4


class M5(M4):
    """Modelo M5 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(M4, self).__init__(*args, **kwargs)
        self.relax()
        self.update()
