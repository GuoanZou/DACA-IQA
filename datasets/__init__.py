"""
Datasets module for DACA-IQA
"""

from .image_dataset import (
    ImageDataset,
    ImageDataset_SPAQ,
    ImageDataset_SPAQWithSoftLabels,
    ImageDataset_TID,
    ImageDataset_PIPAL,
    ImageDataset_ava
)

__all__ = [
    "ImageDataset",
    "ImageDataset_SPAQ",
    "ImageDataset_SPAQWithSoftLabels",
    "ImageDataset_TID",
    "ImageDataset_PIPAL",
    "ImageDataset_ava",
]
