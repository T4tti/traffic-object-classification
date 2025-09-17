"""
Loss functions for object detection training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in object detection.
    
    Reference: "Focal Loss for Dense Object Detection" - Lin et al.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class (typically 0.25)
            gamma: Focusing parameter (typically 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Predictions [N, C] where C = number of classes
            targets: Ground truth class indices [N]

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class GIoULoss(nn.Module):
    """
    Generalized Intersection over Union Loss for bounding box regression.
    
    Reference: "Generalized Intersection over Union" - Rezatofighi et al.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Initialize GIoU Loss.

        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pred_boxes: Predicted boxes [N, 4] in format (x1, y1, x2, y2)
            target_boxes: Target boxes [N, 4] in format (x1, y1, x2, y2)

        Returns:
            GIoU loss value
        """
        # Calculate intersection
        x1_int = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1_int = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2_int = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2_int = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        intersection = torch.clamp(x2_int - x1_int, min=0) * torch.clamp(y2_int - y1_int, min=0)

        # Calculate union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union = pred_area + target_area - intersection

        # Calculate IoU
        iou = intersection / (union + 1e-6)

        # Calculate enclosing box
        x1_enc = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        y1_enc = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        x2_enc = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        y2_enc = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        enclosing_area = (x2_enc - x1_enc) * (y2_enc - y1_enc)

        # Calculate GIoU
        giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)
        
        # GIoU loss
        giou_loss = 1 - giou

        if self.reduction == "mean":
            return giou_loss.mean()
        elif self.reduction == "sum":
            return giou_loss.sum()
        else:
            return giou_loss


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    class_weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute combined loss for object detection.

    Args:
        outputs: Model outputs containing 'logits' and 'boxes'
        targets: List of target dictionaries with 'labels' and 'boxes'
        class_weights: Optional class weights for focal loss

    Returns:
        Combined loss value
    """
    device = outputs["logits"].device
    
    # Flatten predictions and targets
    pred_logits = outputs["logits"].view(-1, outputs["logits"].size(-1))
    pred_boxes = outputs["boxes"].view(-1, 4)
    
    target_labels = []
    target_boxes = []
    
    for target in targets:
        target_labels.append(target["labels"])
        target_boxes.append(target["boxes"])
    
    target_labels = torch.cat(target_labels, dim=0).to(device)
    target_boxes = torch.cat(target_boxes, dim=0).to(device)
    
    # Classification loss (Focal Loss)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    cls_loss = focal_loss(pred_logits, target_labels)
    
    # Regression loss (GIoU Loss)
    # Only compute regression loss for positive samples
    positive_mask = target_labels > 0
    if positive_mask.sum() > 0:
        giou_loss = GIoULoss()
        reg_loss = giou_loss(pred_boxes[positive_mask], target_boxes[positive_mask])
    else:
        reg_loss = torch.tensor(0.0, device=device)
    
    # Combine losses
    total_loss = cls_loss + reg_loss
    
    return total_loss


def compute_balanced_loss(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    class_weights: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute balanced loss for imbalanced datasets.

    Args:
        outputs: Model outputs
        targets: Target annotations
        class_weights: Class weights for balancing
        alpha: Weight for classification loss
        beta: Weight for regression loss

    Returns:
        Total loss and loss components dictionary
    """
    device = outputs["logits"].device
    
    # Flatten predictions and targets
    pred_logits = outputs["logits"].view(-1, outputs["logits"].size(-1))
    pred_boxes = outputs["boxes"].view(-1, 4)
    
    target_labels = []
    target_boxes = []
    
    for target in targets:
        target_labels.append(target["labels"])
        target_boxes.append(target["boxes"])
    
    target_labels = torch.cat(target_labels, dim=0).to(device)
    target_boxes = torch.cat(target_boxes, dim=0).to(device)
    
    # Weighted classification loss
    if class_weights is not None:
        weight_tensor = class_weights.to(device)
        cls_loss = F.cross_entropy(pred_logits, target_labels, weight=weight_tensor)
    else:
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        cls_loss = focal_loss(pred_logits, target_labels)
    
    # Regression loss for positive samples
    positive_mask = target_labels > 0
    if positive_mask.sum() > 0:
        giou_loss = GIoULoss()
        reg_loss = giou_loss(pred_boxes[positive_mask], target_boxes[positive_mask])
    else:
        reg_loss = torch.tensor(0.0, device=device)
    
    # Combine with weights
    total_loss = alpha * cls_loss + beta * reg_loss
    
    loss_dict = {
        "total_loss": total_loss,
        "classification_loss": cls_loss,
        "regression_loss": reg_loss,
    }
    
    return total_loss, loss_dict