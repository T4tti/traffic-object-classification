"""
Metrics computation for object detection evaluation.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any


class ObjectDetectionMetrics:
    """Object detection metrics computation."""

    def __init__(self, class_names: List[str]):
        """
        Initialize metrics calculator.

        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)

    def compute_metrics(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        iou_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for object detection.

        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
            iou_threshold: IoU threshold for positive matches

        Returns:
            Dictionary containing various metrics
        """
        # Compute per-class AP
        per_class_ap = []
        for class_idx in range(self.num_classes):
            ap = self.compute_class_ap(predictions, targets, class_idx, iou_threshold)
            per_class_ap.append(ap)

        # Compute overall metrics
        mAP = np.mean([ap for ap in per_class_ap if not np.isnan(ap)])
        
        # Compute AP at different IoU thresholds
        ap_50 = mAP  # Already computed at 0.5
        ap_75 = np.mean([
            self.compute_class_ap(predictions, targets, class_idx, 0.75)
            for class_idx in range(self.num_classes)
        ])

        # Compute precision, recall, F1
        precision, recall, f1 = self.compute_precision_recall_f1(
            predictions, targets, iou_threshold
        )

        metrics = {
            "mAP": float(mAP),
            "AP50": float(ap_50),
            "AP75": float(ap_75),
            "per_class_ap": per_class_ap,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "class_names": self.class_names,
        }

        return metrics

    def compute_class_ap(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        class_idx: int,
        iou_threshold: float = 0.5,
    ) -> float:
        """
        Compute Average Precision for a specific class.

        Args:
            predictions: List of predictions
            targets: List of targets
            class_idx: Class index to compute AP for
            iou_threshold: IoU threshold

        Returns:
            Average Precision for the class
        """
        # Collect all predictions and ground truths for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        all_gt_matched = []

        for pred, target in zip(predictions, targets):
            # Get predictions for this class
            class_mask = pred["labels"] == class_idx
            if class_mask.sum() > 0:
                class_boxes = pred["boxes"][class_mask]
                class_scores = pred["scores"][class_mask]
                all_pred_boxes.append(class_boxes)
                all_pred_scores.append(class_scores)

            # Get ground truths for this class
            gt_class_mask = target["labels"] == class_idx
            if gt_class_mask.sum() > 0:
                gt_boxes = target["boxes"][gt_class_mask]
                all_gt_boxes.append(gt_boxes)
                all_gt_matched.append(torch.zeros(len(gt_boxes), dtype=torch.bool))

        if not all_pred_boxes or not all_gt_boxes:
            return np.nan

        # Concatenate all predictions and sort by score
        all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
        all_pred_scores = torch.cat(all_pred_scores, dim=0)
        sorted_indices = torch.argsort(all_pred_scores, descending=True)
        
        all_pred_boxes = all_pred_boxes[sorted_indices]
        all_pred_scores = all_pred_scores[sorted_indices]

        # Match predictions to ground truths
        tp = torch.zeros(len(all_pred_boxes))
        fp = torch.zeros(len(all_pred_boxes))
        
        gt_idx = 0
        for i, pred_box in enumerate(all_pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for j, gt_boxes in enumerate(all_gt_boxes):
                ious = compute_iou(pred_box.unsqueeze(0), gt_boxes)
                max_iou, max_idx = torch.max(ious, dim=1)
                
                if max_iou > best_iou:
                    best_iou = max_iou.item()
                    best_gt_idx = (j, max_idx.item())

            # Check if it's a true positive
            if best_iou >= iou_threshold and best_gt_idx != -1:
                gt_list_idx, gt_box_idx = best_gt_idx
                if not all_gt_matched[gt_list_idx][gt_box_idx]:
                    tp[i] = 1
                    all_gt_matched[gt_list_idx][gt_box_idx] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        # Compute precision and recall
        cumsum_tp = torch.cumsum(tp, dim=0)
        cumsum_fp = torch.cumsum(fp, dim=0)
        
        total_gt = sum(len(gt_boxes) for gt_boxes in all_gt_boxes)
        
        precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-8)
        recall = cumsum_tp / (total_gt + 1e-8)

        # Compute AP using 11-point interpolation
        ap = compute_ap(recall.numpy(), precision.numpy())
        return ap

    def compute_precision_recall_f1(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        iou_threshold: float = 0.5,
    ) -> Tuple[float, float, float]:
        """
        Compute overall precision, recall, and F1 score.

        Args:
            predictions: List of predictions
            targets: List of targets
            iou_threshold: IoU threshold

        Returns:
            Tuple of (precision, recall, f1_score)
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, target in zip(predictions, targets):
            # Match predictions to targets
            if len(pred["boxes"]) == 0 and len(target["boxes"]) == 0:
                continue
            elif len(pred["boxes"]) == 0:
                total_fn += len(target["boxes"])
                continue
            elif len(target["boxes"]) == 0:
                total_fp += len(pred["boxes"])
                continue

            # Compute IoU between all predictions and targets
            ious = compute_iou(pred["boxes"], target["boxes"])
            
            # Find best matches
            pred_matched = torch.zeros(len(pred["boxes"]), dtype=torch.bool)
            target_matched = torch.zeros(len(target["boxes"]), dtype=torch.bool)

            # Sort predictions by score
            sorted_indices = torch.argsort(pred["scores"], descending=True)
            
            for pred_idx in sorted_indices:
                best_iou = 0
                best_target_idx = -1
                
                for target_idx in range(len(target["boxes"])):
                    if target_matched[target_idx]:
                        continue
                    
                    iou = ious[pred_idx, target_idx]
                    if iou > best_iou and iou >= iou_threshold:
                        # Check if classes match
                        if pred["labels"][pred_idx] == target["labels"][target_idx]:
                            best_iou = iou
                            best_target_idx = target_idx

                if best_target_idx != -1:
                    pred_matched[pred_idx] = True
                    target_matched[best_target_idx] = True
                    total_tp += 1

            # Count false positives and false negatives
            total_fp += (~pred_matched).sum().item()
            total_fn += (~target_matched).sum().item()

        # Compute metrics
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)

        return precision, recall, f1_score

    def analyze_single_prediction(
        self,
        prediction: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        iou_threshold: float = 0.5,
    ) -> Dict[str, List]:
        """
        Analyze a single prediction for failure cases.

        Args:
            prediction: Single prediction dictionary
            target: Single target dictionary
            iou_threshold: IoU threshold

        Returns:
            Dictionary with false positives, false negatives, and misclassifications
        """
        false_positives = []
        false_negatives = []
        misclassifications = []

        if len(prediction["boxes"]) == 0 and len(target["boxes"]) == 0:
            return {
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "misclassifications": misclassifications,
            }

        if len(prediction["boxes"]) == 0:
            # All targets are false negatives
            for i in range(len(target["boxes"])):
                false_negatives.append({
                    "box": target["boxes"][i].cpu().tolist(),
                    "class": target["labels"][i].item(),
                })
            return {
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "misclassifications": misclassifications,
            }

        if len(target["boxes"]) == 0:
            # All predictions are false positives
            for i in range(len(prediction["boxes"])):
                false_positives.append({
                    "box": prediction["boxes"][i].cpu().tolist(),
                    "class": prediction["labels"][i].item(),
                    "score": prediction["scores"][i].item(),
                })
            return {
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "misclassifications": misclassifications,
            }

        # Compute IoU matrix
        ious = compute_iou(prediction["boxes"], target["boxes"])
        
        # Track matches
        pred_matched = torch.zeros(len(prediction["boxes"]), dtype=torch.bool)
        target_matched = torch.zeros(len(target["boxes"]), dtype=torch.bool)

        # Match predictions to targets
        for pred_idx in range(len(prediction["boxes"])):
            best_iou = 0
            best_target_idx = -1
            
            for target_idx in range(len(target["boxes"])):
                if target_matched[target_idx]:
                    continue
                
                iou = ious[pred_idx, target_idx]
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_target_idx = target_idx

            if best_target_idx != -1:
                pred_class = prediction["labels"][pred_idx].item()
                target_class = target["labels"][best_target_idx].item()
                
                if pred_class == target_class:
                    # Correct detection
                    pred_matched[pred_idx] = True
                    target_matched[best_target_idx] = True
                else:
                    # Misclassification
                    misclassifications.append({
                        "box": prediction["boxes"][pred_idx].cpu().tolist(),
                        "pred_class": pred_class,
                        "true_class": target_class,
                        "score": prediction["scores"][pred_idx].item(),
                        "iou": best_iou,
                    })
                    pred_matched[pred_idx] = True
                    target_matched[best_target_idx] = True

        # Remaining unmatched predictions are false positives
        for pred_idx in range(len(prediction["boxes"])):
            if not pred_matched[pred_idx]:
                false_positives.append({
                    "box": prediction["boxes"][pred_idx].cpu().tolist(),
                    "class": prediction["labels"][pred_idx].item(),
                    "score": prediction["scores"][pred_idx].item(),
                })

        # Remaining unmatched targets are false negatives
        for target_idx in range(len(target["boxes"])):
            if not target_matched[target_idx]:
                false_negatives.append({
                    "box": target["boxes"][target_idx].cpu().tolist(),
                    "class": target["labels"][target_idx].item(),
                })

        return {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "misclassifications": misclassifications,
        }


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: First set of boxes [N, 4]
        boxes2: Second set of boxes [M, 4]

    Returns:
        IoU matrix [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Compute union
    union = area1[:, None] + area2 - intersection

    # Compute IoU
    iou = intersection / (union + 1e-6)
    return iou


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision using 11-point interpolation.

    Args:
        recall: Recall values
        precision: Precision values

    Returns:
        Average Precision
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    
    return ap


def compute_map(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute mean Average Precision (mAP).

    Args:
        predictions: List of predictions
        targets: List of targets
        num_classes: Number of classes
        iou_threshold: IoU threshold

    Returns:
        Mean Average Precision
    """
    metrics_calculator = ObjectDetectionMetrics([f"class_{i}" for i in range(num_classes)])
    metrics = metrics_calculator.compute_metrics(predictions, targets, iou_threshold)
    return metrics["mAP"]