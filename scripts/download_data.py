#!/usr/bin/env python3
"""
Download and prepare datasets for training.
"""

import argparse
import os
import sys
from typing import Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_utils import download_dataset, prepare_annotations, split_dataset, verify_dataset
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=["coco", "cityscapes", "custom"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data",
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=[0.7, 0.2, 0.1],
        help="Train/val/test split ratios"
    )
    parser.add_argument(
        "--custom-urls",
        type=str,
        help="JSON file with custom download URLs"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing dataset"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("dataset_downloader", log_dir=os.path.join(args.data_dir, "logs"))
    
    try:
        if args.verify_only:
            # Verify existing dataset
            dataset_dir = os.path.join(args.data_dir, args.dataset)
            annotations_file = os.path.join(dataset_dir, "annotations.json")
            
            if os.path.exists(annotations_file):
                logger.info(f"Verifying dataset: {args.dataset}")
                stats = verify_dataset(dataset_dir, annotations_file)
                
                logger.info("Dataset Statistics:")
                logger.info(f"  Total images: {stats['total_images']}")
                logger.info(f"  Total objects: {stats['total_objects']}")
                logger.info(f"  Missing images: {stats['missing_images']}")
                logger.info(f"  Empty images: {stats['empty_images']}")
                logger.info(f"  Classes: {list(stats['class_counts'].keys())}")
            else:
                logger.error(f"Annotations file not found: {annotations_file}")
                return 1
                
        else:
            # Download dataset
            logger.info(f"Downloading dataset: {args.dataset}")
            
            urls = None
            if args.custom_urls:
                import json
                with open(args.custom_urls, 'r') as f:
                    urls = json.load(f)
            
            dataset_dir = download_dataset(
                dataset_name=args.dataset,
                data_dir=args.data_dir,
                urls=urls
            )
            
            logger.info(f"Dataset downloaded to: {dataset_dir}")
            
            # Prepare annotations if COCO dataset
            if args.dataset == "coco":
                logger.info("Preparing COCO annotations...")
                
                # Process train annotations
                train_images_dir = os.path.join(dataset_dir, "train_images", "train2017")
                train_annotations = os.path.join(dataset_dir, "annotations", "annotations", "instances_train2017.json")
                train_output = os.path.join(dataset_dir, "train_annotations.json")
                
                if os.path.exists(train_annotations):
                    prepare_annotations(
                        images_dir=train_images_dir,
                        annotations_file=train_annotations,
                        output_file=train_output
                    )
                
                # Process val annotations
                val_images_dir = os.path.join(dataset_dir, "val_images", "val2017")
                val_annotations = os.path.join(dataset_dir, "annotations", "annotations", "instances_val2017.json")
                val_output = os.path.join(dataset_dir, "val_annotations.json")
                
                if os.path.exists(val_annotations):
                    prepare_annotations(
                        images_dir=val_images_dir,
                        annotations_file=val_annotations,
                        output_file=val_output
                    )
            
            # Split dataset if requested
            main_annotations = os.path.join(dataset_dir, "annotations.json")
            if os.path.exists(main_annotations):
                logger.info("Splitting dataset...")
                
                train_ratio, val_ratio, test_ratio = args.split_ratios
                split_files = split_dataset(
                    annotations_file=main_annotations,
                    output_dir=dataset_dir,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio
                )
                
                logger.info(f"Split files created: {split_files}")
            
            logger.info("Dataset preparation completed!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())