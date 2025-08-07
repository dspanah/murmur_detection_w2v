import random

import numpy as np
from packaging.version import parse, Version


def set_global_seed(seed: int) -> None:
    """
    Sets the random seed globally for reproducibility across multiple libraries:
    - PyTorch (CPU and GPU)
    - TensorFlow (version-aware)
    - NumPy
    - Python's built-in `random` module

    Args:
        seed (int): The seed value to set.
    """

    # Try to import PyTorch, and set seed if available
    try:
        import torch
    except ImportError:
        pass  # If PyTorch isn't installed, skip
    else:
        torch.manual_seed(seed)  # Set seed for CPU
        torch.cuda.manual_seed_all(seed)  # Set seed for all CUDA devices (if using GPU)

    # Try to import TensorFlow, and set seed depending on version
    try:
        import tensorflow as tf
    except ImportError:
        pass  # If TensorFlow isn't installed, skip
    else:
        # TensorFlow version >= 2.0.0
        if parse(tf.__version__) >= Version("2.0.0"):
            tf.random.set_seed(seed)
        # TensorFlow version <= 1.13.2 (old API)
        elif parse(tf.__version__) <= Version("1.13.2"):
            tf.set_random_seed(seed)
        # For versions in between, use compat API
        else:
            tf.compat.v1.set_random_seed(seed)

    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)
