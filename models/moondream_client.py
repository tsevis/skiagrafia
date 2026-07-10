"""moondream_client.py  --  Backward-compatibility shim.

The Ollama VLM transport now lives in models.vlm_client together with the
llama.cpp backend. This module keeps the historical import path working:

    from models.moondream_client import MoondreamClient
"""
from __future__ import annotations

from models.vlm_client import (
    MAX_CHILDREN,
    MAX_PARENTS,
    MAX_TOKENS,
    DetectedLabel,
    MoondreamClient,
    OllamaVLMClient,
)

__all__ = [
    "MAX_CHILDREN",
    "MAX_PARENTS",
    "MAX_TOKENS",
    "DetectedLabel",
    "MoondreamClient",
    "OllamaVLMClient",
]
