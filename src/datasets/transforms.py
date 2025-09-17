"""
Data transformation utilities for traffic object detection.
"""

from typing import Any, Dict, List, Optional

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T


def get_transforms(
    phase: str = "train",
    image_size: int = 512,
    augmentation_prob: float = 0.5,
) -> A.Compose:
    """
    Get data transformations for different phases.

    Args:
        phase: Training phase ('train', 'val', 'test')
        image_size: Target image size
        augmentation_prob: Probability for augmentations

    Returns:
        Albumentations composition
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=augmentation_prob),
                A.RandomBrightnessContrast(p=augmentation_prob),
                A.OneOf(
                    [
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ],
                    p=augmentation_prob,
                ),
                A.OneOf(
                    [
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=0.1),
                        A.PiecewiseAffine(p=0.3),
                    ],
                    p=augmentation_prob,
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2),
                        A.Sharpen(),
                        A.Emboss(),
                        A.RandomBrightnessContrast(),
                    ],
                    p=augmentation_prob,
                ),
                A.HueSaturationValue(p=augmentation_prob),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_area=0,
                min_visibility=0,
            ),
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_area=0,
                min_visibility=0,
            ),
        )


def get_torchvision_transforms(phase: str = "train", image_size: int = 512) -> T.Compose:
    """
    Get torchvision transformations (simpler alternative).

    Args:
        phase: Training phase ('train', 'val', 'test')
        image_size: Target image size

    Returns:
        Torchvision composition
    """
    if phase == "train":
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )