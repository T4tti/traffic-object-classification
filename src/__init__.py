"""
Traffic Object Classification Package

This package provides tools and models for traffic object classification
using RetinaNet and Deformable DETR with balanced and imbalanced datasets.
"""

__version__ = "0.1.0"
__author__ = "Lê Nguyễn Thành Tài"
__email__ = "your.email@example.com"

from . import datasets, eval, models, train, utils

__all__ = ["datasets", "eval", "models", "train", "utils"]