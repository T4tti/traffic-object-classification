# Project Setup and Getting Started

## Directory Structure Overview

The project has been set up with a comprehensive directory structure for traffic object classification:

```
traffic-object-classification/
├── src/                          # Source code
│   ├── retinanet/               # RetinaNet model implementation
│   ├── deformable_detr/         # Deformable DETR implementation  
│   └── common/                  # Shared utilities and datasets
├── data/                        # Dataset organization
│   ├── raw/                     # Original images
│   ├── processed/               # Preprocessed images
│   ├── balanced/                # Balanced dataset
│   ├── imbalanced/              # Imbalanced dataset
│   └── annotations/             # Annotation files
├── models/                      # Model storage
│   ├── retinanet/              # RetinaNet checkpoints
│   ├── deformable_detr/        # Deformable DETR checkpoints
│   └── pretrained/             # Pretrained models
├── configs/                     # Configuration files
│   ├── retinanet/              # RetinaNet configs
│   ├── deformable_detr/        # Deformable DETR configs
│   └── data/                   # Data configurations
├── scripts/                     # Training and utility scripts
├── notebooks/                   # Jupyter notebooks for analysis
├── tests/                       # Unit tests
├── docs/                        # Documentation
└── utils/                       # Additional utilities
```

## Next Steps

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Your Dataset**:
   - Place raw images in `data/raw/`
   - Organize balanced dataset in `data/balanced/`
   - Organize imbalanced dataset in `data/imbalanced/`
   - Place annotations in `data/annotations/`

3. **Configure Training**:
   - Modify `configs/retinanet/config.yaml` for RetinaNet training
   - Modify `configs/deformable_detr/config.yaml` for Deformable DETR training
   - Update class definitions in `configs/data/classes.py`

4. **Start Training**:
   ```bash
   # Train RetinaNet
   python scripts/train_retinanet.py --config configs/retinanet/config.yaml
   
   # Train Deformable DETR
   python scripts/train_deformable_detr.py --config configs/deformable_detr/config.yaml
   ```

## Key Features Implemented

- ✅ Complete directory structure for ML project
- ✅ Configuration system with YAML files
- ✅ Modular source code organization
- ✅ Training script templates
- ✅ Dataset organization structure
- ✅ Model storage organization
- ✅ Git ignore for ML projects
- ✅ Requirements file with ML dependencies
- ✅ Documentation and setup guides

## Configuration System

The project uses YAML configuration files that allow you to easily modify:
- Model parameters (architecture, classes, etc.)
- Training hyperparameters (batch size, learning rate, etc.)
- Dataset paths and augmentations
- Logging and experiment tracking settings

## Traffic Object Classes

Default classes are defined in `configs/data/classes.py`:
- car, truck, bus, motorcycle, bicycle
- traffic_light, traffic_sign, pedestrian, road, background

Modify this file to match your specific dataset classes.

## Development Workflow

1. Use the `src/` directory for all source code
2. Store configurations in `configs/` 
3. Place datasets in `data/` subdirectories
4. Save model checkpoints in `models/`
5. Use `notebooks/` for data analysis and experiments
6. Add tests in `tests/` directory

The structure is designed to support both balanced and imbalanced datasets with flexible configuration for different experiment setups.