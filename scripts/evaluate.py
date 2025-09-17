#!/usr/bin/env python3
"""
Evaluation script for traffic object detection models.
"""

import argparse
import os
import sys
import json
from typing import Optional

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from datasets import TrafficDataset, get_transforms
from models import create_model
from eval import Evaluator
from utils.config import load_config
from utils.logger import setup_logger
from utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate traffic object detection model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save individual predictions"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization plots"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output dir if specified
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(config.paths.results_dir, f"evaluation_{args.split}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(
        "evaluator",
        log_level=config.logging.level,
        log_dir=output_dir
    )
    
    logger.info(f"Starting evaluation with configuration: {args.config}")
    logger.info(f"Model checkpoint: {args.checkpoint}")
    logger.info(f"Dataset split: {args.split}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Create model
        logger.info(f"Creating model: {config.model.name}")
        model = create_model(
            model_name=config.model.name,
            num_classes=config.model.num_classes,
            config=config.model,
            pretrained=False  # We'll load from checkpoint
        )
        
        # Load checkpoint
        logger.info("Loading model checkpoint...")
        checkpoint_info = load_checkpoint(
            checkpoint_path=args.checkpoint,
            model=model,
            device=device,
            strict=False
        )
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}")
        
        # Create dataset
        logger.info("Creating dataset...")
        transform = get_transforms(
            phase="test",  # No augmentation for evaluation
            image_size=config.dataset.image_size
        )
        
        dataset = TrafficDataset(
            root=os.path.join(config.dataset.root_dir, "images"),
            annotation_file=os.path.join(config.dataset.root_dir, f"{args.split}_annotations.json"),
            transform=transform,
            class_names=config.classes.names
        )
        
        # Limit samples if specified
        if args.max_samples and args.max_samples < len(dataset):
            indices = list(range(args.max_samples))
            dataset = torch.utils.data.Subset(dataset, indices)
        
        logger.info(f"Evaluation samples: {len(dataset)}")
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=config.dataset.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        # Create evaluator
        evaluator = Evaluator(
            model=model,
            device=device,
            class_names=config.classes.names
        )
        
        # Run evaluation
        logger.info("Running evaluation...")
        metrics = evaluator.evaluate(
            data_loader=data_loader,
            iou_threshold=config.evaluation.iou_threshold,
            confidence_threshold=config.evaluation.confidence_threshold,
            save_predictions=args.save_predictions,
            output_dir=output_dir
        )
        
        # Save metrics
        metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to: {metrics_file}")
        
        # Create visualizations if requested
        if args.visualize:
            logger.info("Creating visualizations...")
            
            # Get some sample predictions for visualization
            model.eval()
            sample_images = []
            sample_predictions = []
            sample_targets = []
            
            with torch.no_grad():
                for i, (images, targets) in enumerate(data_loader):
                    if i >= 5:  # Only visualize first 5 batches
                        break
                    
                    images_device = [img.to(device) for img in images]
                    predictions = model.predict(images_device, score_threshold=0.3)
                    
                    sample_images.extend(images)
                    sample_predictions.extend(predictions)
                    sample_targets.extend(targets)
            
            # Create visualization
            viz_path = os.path.join(output_dir, "sample_predictions.png")
            evaluator.visualize_predictions(
                images=sample_images[:10],
                predictions=sample_predictions[:10],
                targets=sample_targets[:10],
                save_path=viz_path,
                max_images=10
            )
            
            logger.info(f"Visualizations saved to: {viz_path}")
        
        logger.info("Evaluation completed!")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())