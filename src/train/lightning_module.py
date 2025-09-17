"""
PyTorch Lightning module for object detection training.
"""

from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from ..models import create_model
from .losses import compute_loss


class ObjectDetectionModule(pl.LightningModule):
    """PyTorch Lightning module for object detection."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize lightning module.

        Args:
            model_name: Name of the model
            num_classes: Number of classes
            learning_rate: Learning rate
            weight_decay: Weight decay
            optimizer: Optimizer type ('adamw', 'sgd')
            scheduler: Scheduler type ('cosine', 'plateau')
            model_config: Model configuration
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            config=model_config,
            **kwargs
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        """Forward pass."""
        return self.model(images, targets)

    def training_step(self, batch: Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]], batch_idx: int):
        """Training step."""
        images, targets = batch
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'training'):
            # For models that have training mode handling
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        else:
            # Custom loss computation
            outputs = self.model(images)
            losses = compute_loss(outputs, targets)

        self.log("train_loss", losses, prog_bar=True, on_step=True, on_epoch=True)
        self.train_losses.append(losses.item())
        
        return losses

    def validation_step(self, batch: Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]], batch_idx: int):
        """Validation step."""
        images, targets = batch
        
        with torch.no_grad():
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'training'):
                # Set to eval mode for validation
                self.model.eval()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            else:
                outputs = self.model(images)
                losses = compute_loss(outputs, targets)

        self.log("val_loss", losses, prog_bar=True, on_step=False, on_epoch=True)
        self.val_losses.append(losses.item())
        
        return losses

    def test_step(self, batch: Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]], batch_idx: int):
        """Test step."""
        images, targets = batch
        
        # Get predictions
        predictions = self.model.predict(images)
        
        # You can add metrics calculation here
        self.log("test_complete", 1.0)
        
        return {"predictions": predictions, "targets": targets}

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        if self.optimizer_name.lower() == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported")

        if self.scheduler_name.lower() == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=100)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.scheduler_name.lower() == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch"
                }
            }
        else:
            return optimizer

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if self.train_losses:
            avg_loss = sum(self.train_losses) / len(self.train_losses)
            self.log("avg_train_loss", avg_loss)
            self.train_losses.clear()

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if self.val_losses:
            avg_loss = sum(self.val_losses) / len(self.val_losses)
            self.log("avg_val_loss", avg_loss)
            self.val_losses.clear()