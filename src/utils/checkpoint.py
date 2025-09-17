"""
Checkpoint utilities for saving and loading model states.
"""

import os
import torch
from typing import Any, Dict, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    best_metric: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        save_path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        best_metric: Optional best metric value
        metadata: Optional additional metadata
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if best_metric is not None:
        checkpoint['best_metric'] = best_metric
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to map checkpoint to
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Checkpoint information dictionary

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    except RuntimeError as e:
        if strict:
            raise e
        else:
            print(f"Warning: Could not load some model parameters: {e}")
            # Try loading with strict=False
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'best_metric': checkpoint.get('best_metric', None),
        'metadata': checkpoint.get('metadata', {}),
    }
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Resuming from epoch {info['epoch']}")
    
    return info


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        Path to latest checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pth') or f.endswith('.pt')
    ]
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time
    checkpoint_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)),
        reverse=True
    )
    
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
    return latest_checkpoint


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 5,
    keep_best: bool = True,
) -> None:
    """
    Clean up old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to keep the best checkpoint
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pth') or f.endswith('.pt')
    ]
    
    if len(checkpoint_files) <= keep_last_n:
        return
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)),
        reverse=True
    )
    
    # Keep the most recent checkpoints
    files_to_keep = set(checkpoint_files[:keep_last_n])
    
    # Keep best checkpoint if requested
    if keep_best:
        best_checkpoint = find_best_checkpoint(checkpoint_dir)
        if best_checkpoint:
            files_to_keep.add(os.path.basename(best_checkpoint))
    
    # Remove old checkpoints
    for filename in checkpoint_files:
        if filename not in files_to_keep:
            filepath = os.path.join(checkpoint_dir, filename)
            try:
                os.remove(filepath)
                print(f"Removed old checkpoint: {filepath}")
            except Exception as e:
                print(f"Failed to remove {filepath}: {e}")


def find_best_checkpoint(checkpoint_dir: str, metric_name: str = "best_metric") -> Optional[str]:
    """
    Find the checkpoint with the best metric value.

    Args:
        checkpoint_dir: Directory containing checkpoints
        metric_name: Name of the metric to compare

    Returns:
        Path to best checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pth') or f.endswith('.pt')
    ]
    
    if not checkpoint_files:
        return None
    
    best_metric = None
    best_checkpoint = None
    
    for filename in checkpoint_files:
        filepath = os.path.join(checkpoint_dir, filename)
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            if metric_name in checkpoint:
                metric_value = checkpoint[metric_name]
                if best_metric is None or metric_value > best_metric:
                    best_metric = metric_value
                    best_checkpoint = filepath
        except Exception as e:
            print(f"Failed to load checkpoint {filepath}: {e}")
            continue
    
    return best_checkpoint


class CheckpointManager:
    """Manager for handling model checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best: bool = True,
        metric_name: str = "mAP",
        higher_better: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Whether to save best checkpoint separately
            metric_name: Metric name for determining best checkpoint
            higher_better: Whether higher metric values are better
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.metric_name = metric_name
        self.higher_better = higher_better
        
        self.best_metric = None
        self.best_checkpoint_path = None
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> bool:
        """
        Save checkpoint and manage checkpoint files.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            scheduler: Optional scheduler state

        Returns:
            True if this is the best checkpoint so far
        """
        # Regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pth"
        )
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=metrics.get('loss', 0.0),
            save_path=checkpoint_path,
            scheduler=scheduler,
            best_metric=metrics.get(self.metric_name),
            metadata=metrics,
        )
        
        # Check if this is the best checkpoint
        is_best = False
        current_metric = metrics.get(self.metric_name)
        
        if current_metric is not None:
            if self.best_metric is None:
                is_best = True
            elif self.higher_better and current_metric > self.best_metric:
                is_best = True
            elif not self.higher_better and current_metric < self.best_metric:
                is_best = True
            
            if is_best:
                self.best_metric = current_metric
                if self.save_best:
                    best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pth")
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        loss=metrics.get('loss', 0.0),
                        save_path=best_path,
                        scheduler=scheduler,
                        best_metric=current_metric,
                        metadata=metrics,
                    )
                    self.best_checkpoint_path = best_path
        
        # Clean up old checkpoints
        cleanup_old_checkpoints(self.checkpoint_dir, self.max_checkpoints, self.save_best)
        
        return is_best