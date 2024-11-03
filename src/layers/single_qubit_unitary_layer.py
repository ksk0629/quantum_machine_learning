from src.layers.base_learnable_layer import BaseLearnableLayer


class SingleQubitUnitaryLayer(BaseLearnableLayer):
    """SingleQubitUnitaryLayer class, suggested in https://arxiv.org/pdf/2103.11307"""

    def __init__(self, param_prefix: str):
        super().__init__(param_prefix=param_prefix)
