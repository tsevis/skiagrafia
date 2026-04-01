from __future__ import annotations

import logging
import os
import torch

logger = logging.getLogger(__name__)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device() -> torch.device:
    """Return MPS device on Apple Silicon, CPU with warning otherwise."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    logger.warning(
        "MPS unavailable — falling back to CPU. "
        "Performance will be severely degraded on non-Apple Silicon hardware."
    )
    return torch.device("cpu")


DEVICE = get_device()
