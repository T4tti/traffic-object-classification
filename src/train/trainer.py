"""
Training loop implementation for object detection models.
"""

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import compute_loss


class Trainer:
    """Simple trainer for object detection models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "checkpoints",
        **kwargs
    ):
        """
        Initialize trainer.

        Args:
            model: Object detection model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir

        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move data to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'training'):
                # For torchvision models that return loss dict during training
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            else:
                # For custom models
                outputs = self.model(images)
                losses = compute_loss(outputs, targets)

            # Backward pass
            losses.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()

            # Update metrics
            total_loss += losses.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{losses.item():.4f}",
                "avg_loss": f"{total_loss / num_batches:.4f}"
            })

        return total_loss / num_batches

    def validate(self) -> float:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for images, targets in progress_bar:
                # Move data to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'training'):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                else:
                    outputs = self.model(images)
                    losses = compute_loss(outputs, targets)

                total_loss += losses.item()
                num_batches += 1

                progress_bar.set_postfix({
                    "val_loss": f"{losses.item():.4f}",
                    "avg_val_loss": f"{total_loss / num_batches:.4f}"
                })

        return total_loss / num_batches

    def train(self, epochs: int, save_every: int = 10) -> Dict[str, List[float]]:
        """
        Train the model for specified epochs.

        Args:
            epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs

        Returns:
            Training history dictionary
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 30)

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            if self.val_loader is not None:
                self.val_losses.append(val_loss)

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if self.val_loader else train_loss)
                else:
                    self.scheduler.step()

            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}")
            if self.val_loader is not None:
                print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1)

        # Save final checkpoint
        self.save_checkpoint(epochs, is_final=True)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
        }

    def save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if is_final:
            filename = "final_checkpoint.pth"
        else:
            filename = f"checkpoint_epoch_{epoch}.pth"

        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.learning_rates = checkpoint.get("learning_rates", [])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]