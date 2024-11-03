from abc import ABC

from src.layers.base_layer import BaseLayer


class BaseLearnableLayer(ABC, BaseLayer):
    """BaseLearnableLayer abstract class."""

    def __init__(self, param_prefix: str):
        super().__init__()
        self.param_prefix = param_prefix
