"""
Basic utility functions without heavy dependencies.
"""

import yaml
import os
from typing import Dict, Any


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


def setup_directories(base_dir: str):
    """
    Set up necessary directories for training.
    
    Args:
        base_dir: Base directory path
    """
    dirs = ['logs', 'checkpoints', 'results']
    for dir_name in dirs:
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)