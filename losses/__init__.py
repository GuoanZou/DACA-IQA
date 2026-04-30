"""
Losses module for DACA-IQA
"""

from .mnl_loss import (
    kl_rank_loss,
    ordinal_loss,
    Fidelity_Loss,
    Multi_Fidelity_Loss,
    loss_m3,
    loss_m4
)

__all__ = [
    "kl_rank_loss",
    "ordinal_loss",
    "Fidelity_Loss",
    "Multi_Fidelity_Loss",
    "loss_m3",
    "loss_m4",
]
