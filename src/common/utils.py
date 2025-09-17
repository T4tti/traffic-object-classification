"""
Common utility functions for the project.
"""

import yaml
import torch
from typing import Dict, Any
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_model(model: torch.nn.Module, path: str, epoch: int, optimizer_state: Dict = None):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        path: Save path
        epoch: Current epoch
        optimizer_state: Optimizer state dict
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }
    
    if optimizer_state:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    torch.save(checkpoint, path)


def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        path: Checkpoint path
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def setup_directories(base_dir: str):
    """
    Set up necessary directories for training.
    
    Args:
        base_dir: Base directory path
    """
    dirs = ['logs', 'checkpoints', 'results']
    for dir_name in dirs:
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)