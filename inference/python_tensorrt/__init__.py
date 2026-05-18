"""Python TensorRT inference implementation."""

__all__ = ["Model"]


def __getattr__(name):
    if name == "Model":
        from .model import Model

        return Model
    raise AttributeError(name)
