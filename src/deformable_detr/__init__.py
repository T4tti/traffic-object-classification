"""
Deformable DETR module for traffic object classification.
"""

from .model import DeformableDETR
from .train import train_deformable_detr
from .inference import inference_deformable_detr

__all__ = ['DeformableDETR', 'train_deformable_detr', 'inference_deformable_detr']