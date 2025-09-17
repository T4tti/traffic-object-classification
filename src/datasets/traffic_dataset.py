"""
Custom traffic dataset implementation.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    """Custom traffic object detection dataset."""

    def __init__(
        self,
        root: str,
        annotation_file: str,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize traffic dataset.

        Args:
            root: Root directory containing images
            annotation_file: Path to annotation file (JSON format)
            transform: Image transformations
            class_names: List of class names
        """
        self.root = root
        self.transform = transform
        self.class_names = class_names or self._default_classes()

        # Load annotations
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)

        self.image_ids = list(self.annotations.keys())

    def _default_classes(self) -> List[str]:
        """Default traffic object classes."""
        return [
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "person",
            "traffic_light",
            "traffic_sign",
            "stop_sign",
        ]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item from dataset."""
        img_id = self.image_ids[index]
        ann = self.annotations[img_id]

        # Load image
        img_path = os.path.join(self.root, ann["filename"])
        img = Image.open(img_path).convert("RGB")

        # Process annotations
        boxes = []
        labels = []
        for obj in ann["objects"]:
            bbox = obj["bbox"]  # [x, y, w, h]
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]

            class_name = obj["class"]
            if class_name in self.class_names:
                labels.append(self.class_names.index(class_name))
            else:
                labels.append(0)  # Unknown class

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.image_ids)

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset."""
        class_counts = torch.zeros(len(self.class_names))

        for img_id in self.image_ids:
            ann = self.annotations[img_id]
            for obj in ann["objects"]:
                class_name = obj["class"]
                if class_name in self.class_names:
                    class_idx = self.class_names.index(class_name)
                    class_counts[class_idx] += 1

        # Calculate inverse frequency weights
        total_samples = class_counts.sum()
        class_weights = total_samples / (len(self.class_names) * class_counts)
        class_weights[class_counts == 0] = 0  # Set weight to 0 for classes with no samples

        return class_weights