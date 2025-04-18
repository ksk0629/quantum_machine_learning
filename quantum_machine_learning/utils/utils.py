import random

import numpy as np
import torch


class Utils:
    """Utils class"""

    @staticmethod
    def fix_seed(seed: int):
        """Fix the random seeds to have reproducibility.

        :param int seed: random seed
        """
        random.seed(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
