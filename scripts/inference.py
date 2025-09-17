#!/usr/bin/env python3
"""
Inference script for traffic object detection models.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import cv2
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import create_model
from datasets import get_torchvision_transforms
from utils.config import load_config
from utils.checkpoint import load_checkpoint
from utils.logger import setup_logger
from eval.visualizer import Visualizer


def preprocess_image(image_path: str, image_size: int = 512):
    """Preprocess single image for inference."""
    image = Image.open(image_path).convert('RGB')
    
    # Get transforms
    transform = get_torchvision_transforms(phase="test", image_size=image_size)
    
    # Apply transforms
    image_tensor = transform(image)
    
    return image_tensor, image


def postprocess_predictions(predictions, original_size, input_size):
    """Scale predictions back to original image size."""
    orig_h, orig_w = original_size
    input_h, input_w = input_size
    
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h
    
    # Scale bounding boxes
    if len(predictions['boxes']) > 0:
        boxes = predictions['boxes'].clone()
        boxes[:, [0, 2]] *= scale_x  # x coordinates
        boxes[:, [1, 3]] *= scale_y  # y coordinates
        predictions['boxes'] = boxes
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run inference on traffic images")
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
        "--input",
        type=str,
        required=True,
        help="Path to input image or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detection results as text files"
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip visualization and only save detection results"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(
        "inference",
        log_level="INFO",
        log_dir=args.output
    )
    
    logger.info(f"Starting inference with configuration: {args.config}")
    logger.info(f"Model checkpoint: {args.checkpoint}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    
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
            pretrained=False
        )
        
        # Load checkpoint
        logger.info("Loading model checkpoint...")
        load_checkpoint(
            checkpoint_path=args.checkpoint,
            model=model,
            device=device,
            strict=False
        )
        
        model.eval()
        
        # Create visualizer
        visualizer = Visualizer(config.classes.names)
        
        # Get input files
        input_path = Path(args.input)
        if input_path.is_file():
            image_files = [input_path]
        elif input_path.is_dir():
            # Find all image files in directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = [
                f for f in input_path.iterdir()
                if f.suffix.lower() in image_extensions
            ]
        else:
            raise ValueError(f"Input path does not exist: {args.input}")
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        for i, image_path in enumerate(image_files):
            logger.info(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            try:
                # Preprocess image
                image_tensor, original_image = preprocess_image(
                    str(image_path),
                    config.dataset.image_size
                )
                
                # Run inference
                with torch.no_grad():
                    images = [image_tensor.to(device)]
                    predictions = model.predict(
                        images,
                        score_threshold=args.confidence_threshold
                    )[0]  # Get first (and only) prediction
                
                # Postprocess predictions
                original_size = original_image.size[::-1]  # (height, width)
                input_size = (config.dataset.image_size, config.dataset.image_size)
                predictions = postprocess_predictions(
                    predictions, original_size, input_size
                )
                
                # Log detection results
                num_detections = len(predictions['boxes'])
                logger.info(f"  Found {num_detections} detections")
                
                if num_detections > 0:
                    for j in range(num_detections):
                        class_id = predictions['labels'][j].item()
                        score = predictions['scores'][j].item()
                        class_name = config.classes.names[class_id]
                        logger.info(f"    {class_name}: {score:.3f}")
                
                # Save results
                output_name = image_path.stem
                
                # Save detection results as text if requested
                if args.save_txt:
                    txt_path = os.path.join(args.output, f"{output_name}.txt")
                    with open(txt_path, 'w') as f:
                        for j in range(len(predictions['boxes'])):
                            box = predictions['boxes'][j]
                            class_id = predictions['labels'][j].item()
                            score = predictions['scores'][j].item()
                            
                            # Convert to YOLO format (normalized coordinates)
                            x1, y1, x2, y2 = box
                            w, h = original_image.size
                            
                            center_x = (x1 + x2) / 2 / w
                            center_y = (y1 + y2) / 2 / h
                            bbox_w = (x2 - x1) / w
                            bbox_h = (y2 - y1) / h
                            
                            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_w:.6f} {bbox_h:.6f} {score:.6f}\n")
                
                # Create visualization
                if not args.no_visualization:
                    # Convert PIL to numpy array
                    img_array = np.array(original_image)
                    
                    # Create visualization
                    viz_path = os.path.join(args.output, f"{output_name}_detection.jpg")
                    
                    fig = visualizer.visualize_single_image(
                        image_tensor,
                        predictions,
                        save_path=viz_path,
                        confidence_threshold=args.confidence_threshold
                    )
                    
                    # Close figure to free memory
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}")
                continue
        
        logger.info("Inference completed!")
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())