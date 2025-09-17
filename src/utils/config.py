"""
Configuration utilities for loading and managing config files.
"""

import os
from typing import Any, Dict, Optional
import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = OmegaConf.create(config_dict)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing config file {config_path}: {e}")


def save_config(config: DictConfig, save_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object
        save_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        OmegaConf.save(config, f)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations with override taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    return OmegaConf.merge(base_config, override_config)


def create_default_config() -> DictConfig:
    """
    Create a default configuration for traffic object detection.

    Returns:
        Default configuration
    """
    default_config = {
        "model": {
            "name": "retinanet",
            "num_classes": 10,
            "backbone": "resnet50",
            "pretrained": True,
        },
        "dataset": {
            "name": "traffic",
            "root_dir": "data/traffic",
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "image_size": 512,
            "batch_size": 8,
            "num_workers": 4,
        },
        "training": {
            "epochs": 100,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "warmup_epochs": 5,
            "save_every": 10,
        },
        "evaluation": {
            "iou_threshold": 0.5,
            "confidence_threshold": 0.5,
            "metrics": ["mAP", "precision", "recall", "f1"],
        },
        "paths": {
            "data_dir": "data",
            "output_dir": "outputs",
            "checkpoint_dir": "checkpoints",
            "log_dir": "logs",
        },
        "logging": {
            "level": "INFO",
            "use_wandb": False,
            "project_name": "traffic-object-detection",
        },
    }
    
    return OmegaConf.create(default_config)


def validate_config(config: DictConfig) -> bool:
    """
    Validate configuration for required fields.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "model.name",
        "model.num_classes", 
        "dataset.root_dir",
        "training.epochs",
        "training.learning_rate",
    ]
    
    for field in required_fields:
        if not OmegaConf.select(config, field):
            print(f"Missing required config field: {field}")
            return False
    
    return True


def update_config_from_args(config: DictConfig, args: Dict[str, Any]) -> DictConfig:
    """
    Update configuration with command line arguments.

    Args:
        config: Base configuration
        args: Command line arguments dictionary

    Returns:
        Updated configuration
    """
    # Convert args to OmegaConf format
    args_config = OmegaConf.create(args)
    
    # Merge configurations
    updated_config = OmegaConf.merge(config, args_config)
    
    return updated_config