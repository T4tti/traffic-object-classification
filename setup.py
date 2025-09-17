#!/usr/bin/env python3
"""
Setup script for traffic object classification project.
"""

import os
import sys
import subprocess
from pathlib import Path


def create_directories():
    """Create necessary directories for the project."""
    directories = [
        'logs', 'checkpoints', 'results', 'experiments',
        'data/train', 'data/val', 'data/test'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def install_dependencies():
    """Install project dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    return True


def setup_git_hooks():
    """Setup pre-commit hooks for code quality."""
    try:
        subprocess.check_call(["pre-commit", "install"])
        print("Pre-commit hooks installed successfully!")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Could not install pre-commit hooks. Install pre-commit and run 'pre-commit install' manually.")


def main():
    """Main setup function."""
    print("Setting up Traffic Object Classification project...")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("Setup failed due to dependency installation error.")
        sys.exit(1)
    
    # Setup git hooks
    setup_git_hooks()
    
    print("\n" + "="*50)
    print("Setup completed successfully!")
    print("="*50)
    print("\nNext steps:")
    print("1. Place your dataset in the appropriate data/ subdirectories")
    print("2. Review and modify configuration files in configs/")
    print("3. Start training with: python scripts/train_retinanet.py --config configs/retinanet/config.yaml")
    print("4. Or train Deformable DETR with: python scripts/train_deformable_detr.py --config configs/deformable_detr/config.yaml")


if __name__ == "__main__":
    main()