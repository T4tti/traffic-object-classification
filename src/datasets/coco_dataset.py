"""
COCO dataset implementation for traffic object detection.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class COCODataset(Dataset):
    """COCO-style dataset for traffic object detection."""

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize COCO dataset.

        Args:
            root: Root directory of dataset
            annFile: Path to annotation file
            transform: Image transformations
            target_transform: Target transformations
        """
        self.root = root
        self.annFile = annFile
        self.transform = transform
        self.target_transform = target_transform

        # Initialize COCO API
        try:
            from pycocotools.coco import COCO
            self.coco = COCO(annFile)
            self.ids = list(sorted(self.coco.imgs.keys()))
        except ImportError:
            raise ImportError("pycocotools is required for COCO dataset")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item from dataset."""
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Load image
        img_info = coco.loadImgs(img_id)[0]
        path = img_info["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        # Process annotations
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor(img_id)}

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.ids)