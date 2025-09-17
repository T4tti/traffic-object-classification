"""
Backbone networks for object detection models.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter


def get_backbone(
    name: str,
    pretrained: bool = True,
    trainable_layers: int = 3,
    norm_layer: Optional[nn.Module] = None,
) -> nn.Module:
    """
    Get backbone network for object detection.

    Args:
        name: Backbone name ('resnet50', 'resnet101', 'mobilenet_v3')
        pretrained: Whether to use pretrained weights
        trainable_layers: Number of trainable layers from the end
        norm_layer: Normalization layer

    Returns:
        Backbone network
    """
    if name == "resnet50":
        backbone = models.resnet50(pretrained=pretrained, norm_layer=norm_layer)
        return _get_resnet_backbone(backbone, trainable_layers)
    elif name == "resnet101":
        backbone = models.resnet101(pretrained=pretrained, norm_layer=norm_layer)
        return _get_resnet_backbone(backbone, trainable_layers)
    elif name == "mobilenet_v3":
        backbone = models.mobilenet_v3_large(pretrained=pretrained, norm_layer=norm_layer)
        return _get_mobilenet_backbone(backbone, trainable_layers)
    else:
        raise ValueError(f"Backbone {name} not supported")


def _get_resnet_backbone(backbone: nn.Module, trainable_layers: int) -> nn.Module:
    """Process ResNet backbone."""
    # Freeze early layers
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    # Extract feature maps from multiple levels
    return_layers = {
        'layer1': '0',
        'layer2': '1', 
        'layer3': '2',
        'layer4': '3'
    }
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    return backbone


def _get_mobilenet_backbone(backbone: nn.Module, trainable_layers: int) -> nn.Module:
    """Process MobileNet backbone."""
    # Freeze early layers
    total_layers = len(list(backbone.features.children()))
    layers_to_freeze = max(0, total_layers - trainable_layers)
    
    for i, child in enumerate(backbone.features.children()):
        if i < layers_to_freeze:
            for param in child.parameters():
                param.requires_grad_(False)

    # Extract features at multiple scales
    return_layers = {
        '6': '0',   # 1/4 scale
        '12': '1',  # 1/8 scale  
        '16': '2',  # 1/16 scale
    }
    
    backbone.features = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
    return backbone.features


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature extraction.
    """

    def __init__(
        self,
        in_channels_list: list,
        out_channels: int = 256,
        extra_blocks=None,
    ):
        """
        Initialize FPN.

        Args:
            in_channels_list: List of input channels for each level
            out_channels: Output channels for all levels
            extra_blocks: Extra blocks for additional levels
        """
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

        self.extra_blocks = extra_blocks

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through FPN."""
        names = list(x.keys())
        x_list = list(x.values())
        
        last_inner = self.inner_blocks[-1](x_list[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))
        
        for idx in range(len(x_list) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x_list[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = nn.functional.interpolate(
                last_inner, size=feat_shape, mode="nearest"
            )
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))

        if self.extra_blocks is not None:
            results = self.extra_blocks(results, x_list)

        # Convert back to dictionary
        out = {}
        for i, name in enumerate(names):
            out[name] = results[i]
            
        return out