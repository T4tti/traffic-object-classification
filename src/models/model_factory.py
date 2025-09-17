"""
Model factory for creating different object detection models.
"""

from typing import Any, Dict, Optional

from .deformable_detr import DeformableDETR
from .retinanet import RetinaNet


def create_model(
    model_name: str,
    num_classes: int,
    config: Optional[Dict[str, Any]] = None,
    pretrained: bool = True,
    **kwargs
) -> Any:
    """
    Create object detection model.

    Args:
        model_name: Name of the model ('retinanet', 'deformable_detr')
        num_classes: Number of classes
        config: Model configuration dictionary
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model arguments

    Returns:
        Initialized model

    Raises:
        ValueError: If model_name is not supported
    """
    config = config or {}
    
    if model_name.lower() == "retinanet":
        return RetinaNet(
            num_classes=num_classes,
            pretrained=pretrained,
            **config,
            **kwargs
        )
    elif model_name.lower() == "deformable_detr":
        return DeformableDETR(
            num_classes=num_classes,
            **config,
            **kwargs
        )
    else:
        raise ValueError(
            f"Model '{model_name}' not supported. "
            "Available models: 'retinanet', 'deformable_detr'"
        )


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get default configuration for a model.

    Args:
        model_name: Name of the model

    Returns:
        Default configuration dictionary
    """
    configs = {
        "retinanet": {
            "backbone_name": "resnet50",
            "trainable_backbone_layers": 3,
            "min_size": 800,
            "max_size": 1333,
            "score_thresh": 0.05,
            "nms_thresh": 0.5,
            "detections_per_img": 300,
        },
        "deformable_detr": {
            "num_queries": 300,
            "hidden_dim": 256,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "pretrained_model_name": "SenseTime/deformable-detr",
        }
    }
    
    return configs.get(model_name.lower(), {})