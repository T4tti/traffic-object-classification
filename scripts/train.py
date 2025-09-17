#!/usr/bin/env python3
"""
Training script for traffic object detection models.
"""

import argparse
import os
import sys
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from datasets import TrafficDataset, get_transforms
from models import create_model
from train import ObjectDetectionModule
from utils.config import load_config, validate_config
from utils.logger import setup_logger
from torch.utils.data import DataLoader


def create_data_loaders(config):
    """Create data loaders from configuration."""
    # Get transforms
    train_transform = get_transforms(
        phase="train",
        image_size=config.dataset.image_size,
        augmentation_prob=config.dataset.augmentation.get('probability', 0.5)
    )
    val_transform = get_transforms(
        phase="val",
        image_size=config.dataset.image_size
    )
    
    # Create datasets
    train_dataset = TrafficDataset(
        root=os.path.join(config.dataset.root_dir, "images"),
        annotation_file=os.path.join(config.dataset.root_dir, f"{config.dataset.train_split}_annotations.json"),
        transform=train_transform,
        class_names=config.classes.names
    )
    
    val_dataset = TrafficDataset(
        root=os.path.join(config.dataset.root_dir, "images"),
        annotation_file=os.path.join(config.dataset.root_dir, f"{config.dataset.val_split}_annotations.json"),
        transform=val_transform,
        class_names=config.classes.names
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=config.dataset.shuffle,
        num_workers=config.dataset.num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train traffic object detection model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (fast_dev_run)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output dir if specified
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    
    # Validate configuration
    if not validate_config(config):
        print("Invalid configuration!")
        return 1
    
    # Setup directories
    os.makedirs(config.paths.output_dir, exist_ok=True)
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(config.paths.log_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(
        "trainer",
        log_level=config.logging.level,
        log_dir=config.paths.log_dir
    )
    
    logger.info(f"Starting training with configuration: {args.config}")
    logger.info(f"Output directory: {config.paths.output_dir}")
    
    try:
        # Set random seed
        if hasattr(config, 'random_seed'):
            pl.seed_everything(config.random_seed)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(config)
        
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        # Create Lightning module
        logger.info(f"Creating model: {config.model.name}")
        lightning_module = ObjectDetectionModule(
            model_name=config.model.name,
            num_classes=config.model.num_classes,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            optimizer=config.training.optimizer,
            scheduler=config.training.scheduler,
            model_config=config.model
        )
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.paths.checkpoint_dir,
            filename='{epoch:04d}-{val_loss:.4f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            save_last=True,
            every_n_epochs=config.training.save_every
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if hasattr(config.training, 'early_stopping_patience'):
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=config.training.early_stopping_patience,
                mode='min'
            )
            callbacks.append(early_stop_callback)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
        
        # Setup logger
        tb_logger = TensorBoardLogger(
            save_dir=config.paths.log_dir,
            name="tensorboard"
        )
        
        loggers = [tb_logger]
        
        if config.logging.get('use_wandb', False):
            wandb_logger = WandbLogger(
                project=config.logging.wandb_project,
                entity=config.logging.get('wandb_entity'),
                save_dir=config.paths.log_dir
            )
            loggers.append(wandb_logger)
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=config.training.epochs,
            gpus=args.gpus if torch.cuda.is_available() else 0,
            callbacks=callbacks,
            logger=loggers,
            fast_dev_run=args.debug,
            precision=16 if config.get('mixed_precision', False) else 32,
            deterministic=config.get('deterministic', False),
            gradient_clip_val=config.training.get('gradient_clip_norm', 0),
        )
        
        # Resume from checkpoint if specified
        ckpt_path = None
        if args.resume:
            ckpt_path = args.resume
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
        
        # Start training
        logger.info("Starting training...")
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path
        )
        
        logger.info("Training completed!")
        
        # Save final model
        final_model_path = os.path.join(config.paths.output_dir, "final_model.pth")
        torch.save(lightning_module.model.state_dict(), final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())