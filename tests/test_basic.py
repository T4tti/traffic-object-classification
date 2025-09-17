"""
Test configuration and basic model instantiation.
"""

import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.retinanet.model import RetinaNet
from src.deformable_detr.model import DeformableDETR


def test_retinanet_instantiation():
    """Test RetinaNet model can be instantiated."""
    model = RetinaNet(num_classes=10, backbone='resnet50')
    assert model.num_classes == 10
    assert model.backbone == 'resnet50'


def test_deformable_detr_instantiation():
    """Test Deformable DETR model can be instantiated."""
    model = DeformableDETR(num_classes=10, num_queries=100)
    assert model.num_classes == 10
    assert model.num_queries == 100


def test_directory_structure():
    """Test that required directories exist."""
    required_dirs = [
        'src', 'data', 'models', 'configs', 'scripts'
    ]
    
    for dir_name in required_dirs:
        assert os.path.exists(dir_name), f"Directory {dir_name} should exist"


if __name__ == "__main__":
    pytest.main([__file__])