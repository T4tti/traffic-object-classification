"""
Deformable DETR implementation for traffic object detection.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection


class DeformableDETR(nn.Module):
    """
    Deformable DETR model for traffic object detection.
    
    Based on the paper: "Deformable DETR: Deformable Transformers for End-to-End Object Detection"
    """

    def __init__(
        self,
        num_classes: int,
        num_queries: int = 300,
        hidden_dim: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        pretrained_model_name: Optional[str] = "SenseTime/deformable-detr",
        **kwargs
    ):
        """
        Initialize Deformable DETR model.

        Args:
            num_classes: Number of object classes
            num_queries: Number of object queries
            hidden_dim: Hidden dimension size
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            pretrained_model_name: HuggingFace model name for pretrained weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        if pretrained_model_name:
            # Load pretrained model and adapt for custom classes
            self.model = DeformableDetrForObjectDetection.from_pretrained(
                pretrained_model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # Create model from scratch
            config = DeformableDetrConfig(
                num_labels=num_classes,
                num_queries=num_queries,
                d_model=hidden_dim,
                encoder_layers=num_encoder_layers,
                decoder_layers=num_decoder_layers,
                **kwargs
            )
            self.model = DeformableDetrForObjectDetection(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
        labels: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        Forward pass.

        Args:
            pixel_values: Input images tensor [batch_size, 3, height, width]
            pixel_mask: Pixel mask tensor
            labels: Target labels for training

        Returns:
            Model outputs including logits and bounding boxes
        """
        return self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )

    def predict(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
        score_threshold: float = 0.5,
        target_sizes: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predict objects in images.

        Args:
            pixel_values: Input images
            pixel_mask: Pixel mask
            score_threshold: Minimum confidence score
            target_sizes: Target image sizes for post-processing

        Returns:
            List of predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            
        # Post-process predictions
        if target_sizes is not None:
            # Use model's post-processor if available
            if hasattr(self.model, 'post_process_object_detection'):
                predictions = self.model.post_process_object_detection(
                    outputs, 
                    target_sizes=target_sizes,
                    threshold=score_threshold
                )
            else:
                predictions = self._post_process_predictions(
                    outputs, target_sizes, score_threshold
                )
        else:
            predictions = self._post_process_predictions(
                outputs, None, score_threshold
            )
            
        return predictions

    def _post_process_predictions(
        self,
        outputs,
        target_sizes: Optional[torch.Tensor] = None,
        score_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process model predictions."""
        logits = outputs.logits
        bboxes = outputs.pred_boxes
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Get max probability and corresponding class for each query
        scores, labels = probs.max(dim=-1)
        
        predictions = []
        for i in range(logits.shape[0]):  # batch size
            # Filter by score threshold
            mask = scores[i] > score_threshold
            
            pred = {
                "scores": scores[i][mask],
                "labels": labels[i][mask],
                "boxes": bboxes[i][mask],
            }
            
            # Convert boxes to absolute coordinates if target sizes provided
            if target_sizes is not None:
                h, w = target_sizes[i]
                boxes = pred["boxes"]
                # Convert from [cx, cy, w, h] normalized to [x1, y1, x2, y2] absolute
                cx, cy, box_w, box_h = boxes.unbind(-1)
                x1 = (cx - 0.5 * box_w) * w
                y1 = (cy - 0.5 * box_h) * h
                x2 = (cx + 0.5 * box_w) * w
                y2 = (cy + 0.5 * box_h) * h
                pred["boxes"] = torch.stack([x1, y1, x2, y2], dim=-1)
                
            predictions.append(pred)
            
        return predictions