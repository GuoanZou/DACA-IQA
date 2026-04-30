"""
DACA-IQA: Degradation-Aware CLIP for Image Quality Assessment
"""

from .models.daca_iqa import DACA_IQA
from .models.dpim import DPIM
from .models.clip_with_cmma import load as load_clip_with_cmma

__version__ = "1.0.0"

__all__ = [
    "DACA_IQA",
    "DPIM",
    "load_clip_with_cmma",
]
