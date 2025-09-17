"""
Logging utilities for traffic object detection.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "traffic_detection",
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_level: Logging level
        log_dir: Directory to save log files
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{name}_{timestamp}.log"
        log_filepath = os.path.join(log_dir, log_filename)
        
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_filepath}")

    return logger


def get_logger(name: str = "traffic_detection") -> logging.Logger:
    """
    Get existing logger or create a basic one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # If no handlers exist, create a basic console logger
        logger = setup_logger(name, log_to_file=False)
    
    return logger


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


def log_model_info(logger: logging.Logger, model, dataset_size: Optional[int] = None):
    """
    Log model information.

    Args:
        logger: Logger instance
        model: Model to log info about
        dataset_size: Optional dataset size
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Model Information:")
    logger.info(f"  Model type: {type(model).__name__}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    if dataset_size:
        logger.info(f"  Dataset size: {dataset_size:,}")


def log_training_config(logger: logging.Logger, config):
    """
    Log training configuration.

    Args:
        logger: Logger instance
        config: Configuration object
    """
    logger.info("Training Configuration:")
    
    if hasattr(config, 'model'):
        logger.info(f"  Model: {config.model.name}")
        logger.info(f"  Backbone: {getattr(config.model, 'backbone', 'N/A')}")
        logger.info(f"  Number of classes: {config.model.num_classes}")
    
    if hasattr(config, 'training'):
        logger.info(f"  Epochs: {config.training.epochs}")
        logger.info(f"  Learning rate: {config.training.learning_rate}")
        logger.info(f"  Batch size: {getattr(config.dataset, 'batch_size', 'N/A')}")
        logger.info(f"  Optimizer: {config.training.optimizer}")
    
    if hasattr(config, 'dataset'):
        logger.info(f"  Dataset: {config.dataset.name}")
        logger.info(f"  Image size: {config.dataset.image_size}")


def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = ""):
    """
    Log evaluation metrics.

    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        prefix: Optional prefix for log messages
    """
    if prefix:
        prefix = f"{prefix} "
    
    logger.info(f"{prefix}Evaluation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
            logger.info(f"  {key}: {[f'{v:.4f}' for v in value]}")
        else:
            logger.info(f"  {key}: {value}")


class WandbLogger:
    """Weights & Biases logger wrapper."""
    
    def __init__(self, project_name: str, config: dict, enabled: bool = True):
        """
        Initialize W&B logger.

        Args:
            project_name: W&B project name
            config: Configuration dictionary
            enabled: Whether to enable W&B logging
        """
        self.enabled = enabled
        
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
                self.run = wandb.init(
                    project=project_name,
                    config=config
                )
            except ImportError:
                print("Warning: wandb not installed. Disabling W&B logging.")
                self.enabled = False
    
    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to W&B."""
        if self.enabled and hasattr(self, 'wandb'):
            self.wandb.log(metrics, step=step)
    
    def log_image(self, image, caption: str = ""):
        """Log image to W&B."""
        if self.enabled and hasattr(self, 'wandb'):
            self.wandb.log({"image": self.wandb.Image(image, caption=caption)})
    
    def finish(self):
        """Finish W&B run."""
        if self.enabled and hasattr(self, 'wandb'):
            self.wandb.finish()