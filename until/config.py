import random

import numpy as np
import torch


def setup_seed(seed):
    """
    设置随机种子参数，使每次结果都一样，可复现
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

