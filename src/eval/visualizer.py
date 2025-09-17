"""
Visualization utilities for traffic object detection.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple, Any
import seaborn as sns


class Visualizer:
    """Visualization utilities for object detection."""

    def __init__(self, class_names: List[str]):
        """
        Initialize visualizer.

        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.colors = self._generate_colors(len(class_names))

    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each class."""
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            rgb = plt.cm.hsv(hue)[:3]
            colors.append(tuple(int(255 * c) for c in rgb))
        return colors

    def visualize_batch(
        self,
        images: List[torch.Tensor],
        predictions: List[Dict[str, torch.Tensor]],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        save_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ) -> plt.Figure:
        """
        Visualize a batch of images with predictions and targets.

        Args:
            images: List of input images
            predictions: List of model predictions
            targets: Optional ground truth targets
            save_path: Optional path to save the figure
            confidence_threshold: Minimum confidence to display

        Returns:
            Matplotlib figure
        """
        batch_size = len(images)
        cols = min(4, batch_size)
        rows = (batch_size + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        if batch_size == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i in range(batch_size):
            ax = axes[i]
            
            # Convert tensor to PIL image
            img = self._tensor_to_pil(images[i])
            ax.imshow(img)
            ax.set_title(f"Image {i}")
            ax.axis('off')

            # Draw predictions
            pred = predictions[i]
            self._draw_boxes_matplotlib(
                ax, pred["boxes"], pred["labels"], pred["scores"],
                confidence_threshold, box_type="pred"
            )

            # Draw targets if available
            if targets is not None:
                target = targets[i]
                self._draw_boxes_matplotlib(
                    ax, target["boxes"], target["labels"], None,
                    0.0, box_type="gt"
                )

        # Hide empty subplots
        for i in range(batch_size, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def visualize_single_image(
        self,
        image: torch.Tensor,
        prediction: Dict[str, torch.Tensor],
        target: Optional[Dict[str, torch.Tensor]] = None,
        save_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ) -> plt.Figure:
        """
        Visualize a single image with predictions and targets.

        Args:
            image: Input image tensor
            prediction: Model prediction
            target: Optional ground truth target
            save_path: Optional path to save the figure
            confidence_threshold: Minimum confidence to display

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Convert tensor to PIL image
        img = self._tensor_to_pil(image)
        ax.imshow(img)
        ax.set_title("Object Detection Results")
        ax.axis('off')

        # Draw predictions
        self._draw_boxes_matplotlib(
            ax, prediction["boxes"], prediction["labels"], prediction["scores"],
            confidence_threshold, box_type="pred"
        )

        # Draw targets if available
        if target is not None:
            self._draw_boxes_matplotlib(
                ax, target["boxes"], target["labels"], None,
                0.0, box_type="gt"
            )

        # Add legend
        self._add_legend(ax)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL image."""
        if tensor.dim() == 3:
            # Denormalize if needed (assuming ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            # Check if tensor is normalized
            if tensor.min() < 0:
                tensor = tensor * std + mean
            
            tensor = torch.clamp(tensor, 0, 1)
            
            # Convert to PIL
            tensor = (tensor * 255).byte()
            tensor = tensor.permute(1, 2, 0)
            img = Image.fromarray(tensor.numpy())
        else:
            raise ValueError("Expected 3D tensor (C, H, W)")
        
        return img

    def _draw_boxes_matplotlib(
        self,
        ax: plt.Axes,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: Optional[torch.Tensor],
        confidence_threshold: float,
        box_type: str = "pred",
    ):
        """Draw bounding boxes on matplotlib axes."""
        if len(boxes) == 0:
            return

        for i, (box, label) in enumerate(zip(boxes, labels)):
            if scores is not None and scores[i] < confidence_threshold:
                continue

            x1, y1, x2, y2 = box.cpu().numpy()
            width = x2 - x1
            height = y2 - y1

            # Choose color and style based on box type
            if box_type == "pred":
                color = self.colors[label.item() % len(self.colors)]
                color = tuple(c / 255.0 for c in color)  # Normalize for matplotlib
                linestyle = '-'
                alpha = 0.8
            else:  # ground truth
                color = 'lime'
                linestyle = '--'
                alpha = 0.6

            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=color, facecolor='none',
                linestyle=linestyle, alpha=alpha
            )
            ax.add_patch(rect)

            # Add label text
            class_name = self.class_names[label.item()]
            if scores is not None:
                text = f"{class_name}: {scores[i]:.2f}"
            else:
                text = class_name

            ax.text(
                x1, y1 - 5, text,
                fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7)
            )

    def _add_legend(self, ax: plt.Axes):
        """Add legend to distinguish predictions from ground truth."""
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Predictions'),
            Line2D([0], [0], color='lime', lw=2, linestyle='--', label='Ground Truth')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    def create_class_distribution_plot(
        self,
        class_counts: Dict[str, int],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a bar plot of class distribution.

        Args:
            class_counts: Dictionary of class names and their counts
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = ax.bar(classes, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution in Dataset')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    learning_rates: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training curves for loss and learning rate.

    Args:
        train_losses: Training loss values
        val_losses: Validation loss values
        learning_rates: Optional learning rate values
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    if learning_rates:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs[:len(val_losses)], val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot learning rate if available
    if learning_rates:
        ax2.plot(epochs[:len(learning_rates)], learning_rates, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Optional path to save the figure
        normalize: Whether to normalize the matrix

    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        cm = confusion_matrix
        title = 'Confusion Matrix'
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig