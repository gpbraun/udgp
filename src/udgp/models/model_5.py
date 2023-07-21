"""Gabriel Braun, 2023

Este módulo implementa o modelo M5 para instâncias do problema uDGP.
"""

from .model_4 import M4


class M5(M4):
    """Modelo M5 para o uDGP."""

    def __init__(self, *args, **kwargs):
        super(M5, self).__init__(*args, **kwargs)
        self.name = "uDGP-M5"
        self.relax()
        self.update()
