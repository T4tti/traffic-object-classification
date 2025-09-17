"""
Visualization utilities for training and evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd


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


def plot_class_distribution(
    class_counts: Dict[str, int],
    save_path: Optional[str] = None,
    top_n: Optional[int] = None,
) -> plt.Figure:
    """
    Plot class distribution as a bar chart.

    Args:
        class_counts: Dictionary of class names and their counts
        save_path: Optional path to save the figure
        top_n: Optional number of top classes to show

    Returns:
        Matplotlib figure
    """
    # Sort by count
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    if top_n:
        sorted_items = sorted_items[:top_n]
    
    classes, counts = zip(*sorted_items)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
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


def plot_metrics_comparison(
    metrics_data: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot comparison of metrics across different models/experiments.

    Args:
        metrics_data: Dictionary with experiment names as keys and metrics as values
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(metrics_data).T
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(numeric_cols[:4]):  # Plot up to 4 metrics
        ax = axes[i]
        
        values = df[metric].values
        experiments = df.index.tolist()
        
        bars = ax.bar(experiments, values, alpha=0.7)
        ax.set_title(f'{metric.upper()}')
        ax.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Hide unused subplots
    for i in range(len(numeric_cols), 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_detection_results(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: List[str],
    confidence_threshold: float = 0.5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot detection results on an image.

    Args:
        image: Input image array
        boxes: Bounding boxes [N, 4] in format (x1, y1, x2, y2)
        labels: Class labels [N]
        scores: Confidence scores [N]
        class_names: List of class names
        confidence_threshold: Minimum confidence to display
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.imshow(image)
    ax.set_title('Object Detection Results')
    ax.axis('off')
    
    # Filter by confidence threshold
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Generate colors for each class
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        color = colors[label % len(colors)]
        
        # Draw rectangle
        rect = plt.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label text
        class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
        text = f"{class_name}: {score:.2f}"
        
        ax.text(
            x1, y1 - 5, text,
            fontsize=10, color='white',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7)
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    ap_score: float,
    class_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot precision-recall curve.

    Args:
        precision: Precision values
        recall: Recall values
        ap_score: Average Precision score
        class_name: Name of the class
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.plot(recall, precision, linewidth=2, label=f'AP = {ap_score:.3f}')
    ax.fill_between(recall, precision, alpha=0.3)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve {class_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_training_dashboard(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a comprehensive training dashboard.

    Args:
        metrics_history: Dictionary containing training metrics history
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
        epochs = range(1, len(metrics_history['train_loss']) + 1)
        axes[0, 0].plot(epochs, metrics_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, metrics_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # mAP plot
    if 'train_map' in metrics_history and 'val_map' in metrics_history:
        epochs = range(1, len(metrics_history['train_map']) + 1)
        axes[0, 1].plot(epochs, metrics_history['train_map'], 'b-', label='Train mAP')
        axes[0, 1].plot(epochs, metrics_history['val_map'], 'r-', label='Val mAP')
        axes[0, 1].set_title('Mean Average Precision')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    if 'learning_rate' in metrics_history:
        epochs = range(1, len(metrics_history['learning_rate']) + 1)
        axes[1, 0].plot(epochs, metrics_history['learning_rate'], 'g-')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Combined metrics plot
    metric_names = ['precision', 'recall', 'f1_score']
    for metric_name in metric_names:
        if metric_name in metrics_history:
            epochs = range(1, len(metrics_history[metric_name]) + 1)
            axes[1, 1].plot(epochs, metrics_history[metric_name], label=metric_name.capitalize())
    
    axes[1, 1].set_title('Evaluation Metrics')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig