"""
RetinaNet implementation for traffic object detection.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead


class RetinaNet(nn.Module):
    """
    RetinaNet model for traffic object detection.
    
    Based on the paper: "Focal Loss for Dense Object Detection"
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        trainable_backbone_layers: int = 3,
        **kwargs
    ):
        """
        Initialize RetinaNet model.

        Args:
            num_classes: Number of object classes (including background)
            backbone_name: Backbone architecture name
            pretrained: Whether to use pretrained weights
            trainable_backbone_layers: Number of trainable backbone layers
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        if backbone_name == "resnet50":
            self.model = retinanet_resnet50_fpn(
                pretrained=pretrained,
                num_classes=num_classes,
                trainable_backbone_layers=trainable_backbone_layers,
                **kwargs
            )
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        """
        Forward pass.

        Args:
            images: List of input images
            targets: List of target dictionaries (for training)

        Returns:
            Predictions or losses depending on mode
        """
        return self.model(images, targets)

    def predict(self, images: List[torch.Tensor], score_threshold: float = 0.5) -> List[Dict[str, torch.Tensor]]:
        """
        Predict objects in images.

        Args:
            images: List of input images
            score_threshold: Minimum confidence score

        Returns:
            List of predictions
        """
        self.eval()
        with torch.no_grad():
            predictions = self.model(images)
            
        # Filter predictions by score threshold
        filtered_predictions = []
        for pred in predictions:
            mask = pred["scores"] > score_threshold
            filtered_pred = {
                "boxes": pred["boxes"][mask],
                "labels": pred["labels"][mask],
                "scores": pred["scores"][mask],
            }
            filtered_predictions.append(filtered_pred)
            
        return filtered_predictions

    def get_backbone_features(self, images: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract backbone features."""
        self.eval()
        with torch.no_grad():
            features = self.model.backbone(images)
        return features