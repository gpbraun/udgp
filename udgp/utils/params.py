"""
Este módulo implementa funções para manipulação de parâmetros.

Gabriel Braun, 2025
"""

__all__ = [
    "ParamView",
]


class ParamView:
    """
    Read/write façade over the instance-level `_model_params` dictionary.
    """

    __slots__ = ("_store",)

    @staticmethod
    def _norm(param_name: str) -> str:
        return param_name.replace("_", "").lower()

    def __init__(self, store: dict[str, float]):
        object.__setattr__(self, "_store", {self._norm(k): v for k, v in store.items()})

    def __dir__(self) -> list[str]:
        """
        `dir()` support.
        """
        return list(self._store)

    def __getattr__(self, param_name: str) -> float:
        """
        Attribute *read* → look up in the internal store.
        """
        try:
            return self._store[self._norm(param_name)]
        except KeyError as exc:
            raise AttributeError(f"Unknown model parameter {param_name!r}") from exc

    def __setattr__(self, param_name: str, value: float) -> None:
        """
        Attribute *write* → update the store.
        """
        self._store[self._norm(param_name)] = value
