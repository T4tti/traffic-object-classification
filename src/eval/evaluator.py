"""
Model evaluation utilities.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import ObjectDetectionMetrics
from .visualizer import Visualizer


class Evaluator:
    """Evaluator for object detection models."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: Object detection model
            device: Device to run evaluation on
            class_names: List of class names
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names or self._default_classes()
        
        self.metrics = ObjectDetectionMetrics(self.class_names)
        self.visualizer = Visualizer(self.class_names)

    def _default_classes(self) -> List[str]:
        """Default traffic object classes."""
        return [
            "background",
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

    def evaluate(
        self,
        data_loader: DataLoader,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        save_predictions: bool = False,
        output_dir: str = "evaluation_results",
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset.

        Args:
            data_loader: Data loader for evaluation
            iou_threshold: IoU threshold for AP calculation
            confidence_threshold: Confidence threshold for predictions
            save_predictions: Whether to save predictions
            output_dir: Output directory for results

        Returns:
            Evaluation metrics dictionary
        """
        self.model.eval()
        
        if save_predictions:
            os.makedirs(output_dir, exist_ok=True)
        
        all_predictions = []
        all_targets = []
        
        print("Running evaluation...")
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(data_loader)):
                # Move to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Get predictions
                predictions = self.model.predict(images, score_threshold=confidence_threshold)
                
                # Store results
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                
                # Save individual predictions if requested
                if save_predictions:
                    for i, (pred, target) in enumerate(zip(predictions, targets)):
                        result = {
                            "batch_idx": batch_idx,
                            "image_idx": i,
                            "prediction": {
                                "boxes": pred["boxes"].cpu().tolist(),
                                "labels": pred["labels"].cpu().tolist(), 
                                "scores": pred["scores"].cpu().tolist(),
                            },
                            "target": {
                                "boxes": target["boxes"].cpu().tolist(),
                                "labels": target["labels"].cpu().tolist(),
                            }
                        }
                        
                        filename = f"prediction_{batch_idx:04d}_{i:04d}.json"
                        with open(os.path.join(output_dir, filename), "w") as f:
                            json.dump(result, f, indent=2)
        
        # Compute metrics
        metrics = self.metrics.compute_metrics(
            all_predictions,
            all_targets,
            iou_threshold=iou_threshold
        )
        
        # Print results
        self._print_results(metrics)
        
        # Save metrics
        if save_predictions:
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        
        return metrics

    def _print_results(self, metrics: Dict[str, Any]):
        """Print evaluation results."""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print(f"Mean Average Precision (mAP): {metrics['mAP']:.4f}")
        print(f"Average Precision (AP@0.5): {metrics['AP50']:.4f}")
        print(f"Average Precision (AP@0.75): {metrics['AP75']:.4f}")
        
        print("\nPer-class Average Precision:")
        for i, class_name in enumerate(self.class_names):
            if i < len(metrics['per_class_ap']):
                ap = metrics['per_class_ap'][i]
                print(f"  {class_name}: {ap:.4f}")
        
        print(f"\nPrecision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")

    def visualize_predictions(
        self,
        images: List[torch.Tensor],
        predictions: List[Dict[str, torch.Tensor]],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        save_path: Optional[str] = None,
        max_images: int = 10,
    ):
        """
        Visualize predictions on images.

        Args:
            images: List of input images
            predictions: List of model predictions
            targets: Optional ground truth targets
            save_path: Path to save visualizations
            max_images: Maximum number of images to visualize
        """
        return self.visualizer.visualize_batch(
            images[:max_images],
            predictions[:max_images],
            targets[:max_images] if targets else None,
            save_path=save_path
        )

    def compare_models(
        self,
        model_predictions: Dict[str, List[Dict[str, torch.Tensor]]],
        targets: List[Dict[str, torch.Tensor]],
        iou_threshold: float = 0.5,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models' predictions.

        Args:
            model_predictions: Dictionary of model name -> predictions
            targets: Ground truth targets
            iou_threshold: IoU threshold for evaluation

        Returns:
            Comparison results dictionary
        """
        results = {}
        
        for model_name, predictions in model_predictions.items():
            metrics = self.metrics.compute_metrics(
                predictions,
                targets,
                iou_threshold=iou_threshold
            )
            results[model_name] = metrics
        
        # Print comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(f"{'Model':<20} {'mAP':<8} {'AP50':<8} {'AP75':<8} {'Precision':<10} {'Recall':<8}")
        print("-"*60)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<20} {metrics['mAP']:<8.4f} {metrics['AP50']:<8.4f} "
                  f"{metrics['AP75']:<8.4f} {metrics['precision']:<10.4f} {metrics['recall']:<8.4f}")
        
        return results

    def analyze_failures(
        self,
        data_loader: DataLoader,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        max_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Analyze model failures and common error patterns.

        Args:
            data_loader: Data loader for analysis
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold
            max_samples: Maximum samples to analyze

        Returns:
            Failure analysis results
        """
        self.model.eval()
        
        false_positives = []
        false_negatives = []
        misclassifications = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(data_loader)):
                if batch_idx >= max_samples:
                    break
                    
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                predictions = self.model.predict(images, score_threshold=confidence_threshold)
                
                # Analyze each image in batch
                for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
                    analysis = self.metrics.analyze_single_prediction(
                        pred, target, iou_threshold
                    )
                    
                    if analysis["false_positives"]:
                        false_positives.extend(analysis["false_positives"])
                    if analysis["false_negatives"]:
                        false_negatives.extend(analysis["false_negatives"])
                    if analysis["misclassifications"]:
                        misclassifications.extend(analysis["misclassifications"])
        
        # Summarize failures
        failure_summary = {
            "total_false_positives": len(false_positives),
            "total_false_negatives": len(false_negatives),
            "total_misclassifications": len(misclassifications),
            "false_positive_classes": self._count_class_occurrences(false_positives),
            "false_negative_classes": self._count_class_occurrences(false_negatives),
            "misclassification_patterns": self._analyze_misclassifications(misclassifications),
        }
        
        return failure_summary

    def _count_class_occurrences(self, detections: List[Dict]) -> Dict[str, int]:
        """Count occurrences by class."""
        counts = {}
        for det in detections:
            class_idx = det.get("class", 0)
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else "unknown"
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def _analyze_misclassifications(self, misclassifications: List[Dict]) -> Dict[str, Dict[str, int]]:
        """Analyze misclassification patterns."""
        patterns = {}
        for misc in misclassifications:
            true_class = self.class_names[misc["true_class"]]
            pred_class = self.class_names[misc["pred_class"]]
            
            if true_class not in patterns:
                patterns[true_class] = {}
            patterns[true_class][pred_class] = patterns[true_class].get(pred_class, 0) + 1
        
        return patterns