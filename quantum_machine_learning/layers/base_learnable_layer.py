from abc import ABC

from quantum_machine_learning.layers.base_layer import BaseLayer


class BaseLearnableLayer(BaseLayer, ABC):
    """BaseLearnableLayer abstract class."""

    def __init__(self, param_prefix: str):
        super().__init__()
        self.param_prefix = param_prefix
