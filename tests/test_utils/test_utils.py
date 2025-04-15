import random

import numpy as np
import pytest
import torch

from quantum_machine_learning.utils.utils import Utils


class TestUtils:

    @classmethod
    def setup_class(cls):
        cls.seed = 901

    def test_fix_seed_with_self_args(self):
        """Normal test;
        Run fix_seed and generate random integers through each module and do the same thing.

        Check if
        1. the generated integers are the same.
        """
        low = 0
        high = 100000

        # 1. the generated integers are the same.
        Utils.fix_seed(self.seed)
        x_random = random.randint(low, high)
        x_np = np.random.randint(low, high)
        x_torch = torch.randint(low=low, high=high, size=(1,))

        Utils.fix_seed(self.seed)
        assert x_random == random.randint(low, high)
        assert x_np == np.random.randint(low, high)
        assert x_torch == torch.randint(low=low, high=high, size=(1,))
