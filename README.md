# Traffic Object Classification

Multi-model traffic object classification using RetinaNet and Deformable DETR with balanced and imbalanced datasets.

## Project Structure

```
traffic-object-classification/
├── src/                           # Source code
│   ├── retinanet/                # RetinaNet implementation
│   ├── deformable_detr/          # Deformable DETR implementation
│   └── common/                   # Shared utilities
├── data/                         # Dataset storage
│   ├── raw/                      # Original dataset
│   ├── processed/                # Preprocessed data
│   ├── balanced/                 # Balanced dataset
│   ├── imbalanced/              # Imbalanced dataset
│   └── annotations/             # Annotation files
├── models/                       # Model storage
│   ├── retinanet/               # RetinaNet checkpoints
│   ├── deformable_detr/         # Deformable DETR checkpoints
│   └── pretrained/              # Pretrained models
├── configs/                      # Configuration files
│   ├── retinanet/               # RetinaNet configs
│   ├── deformable_detr/         # Deformable DETR configs
│   └── data/                    # Data configs
├── scripts/                      # Training and utility scripts
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
├── docs/                         # Documentation
└── utils/                        # Utility scripts
```

## Features

- **Multi-model approach**: Support for both RetinaNet and Deformable DETR architectures
- **Dataset flexibility**: Handles both balanced and imbalanced traffic datasets
- **Configurable training**: YAML-based configuration system
- **Experiment tracking**: Integration with Weights & Biases and TensorBoard
- **Comprehensive evaluation**: Multiple metrics for model assessment

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/T4tti/traffic-object-classification.git
cd traffic-object-classification

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

Place your traffic object dataset in the appropriate directories:
- Raw images: `data/raw/`
- Balanced dataset: `data/balanced/`
- Imbalanced dataset: `data/imbalanced/`
- Annotations: `data/annotations/`

### 3. Training

#### RetinaNet
```bash
python scripts/train_retinanet.py --config configs/retinanet/config.yaml
```

#### Deformable DETR
```bash
python scripts/train_deformable_detr.py --config configs/deformable_detr/config.yaml
```

## Configuration

Edit the configuration files in `configs/` to customize:
- Model architecture parameters
- Training hyperparameters
- Dataset paths and augmentations
- Logging and experiment tracking

## Dataset Format

The project supports multiple annotation formats:
- COCO format (recommended)
- YOLO format
- XML format

## Models

### RetinaNet
- Focal loss for handling class imbalance
- Feature Pyramid Network (FPN)
- Anchor-based detection

### Deformable DETR
- Deformable attention mechanism
- End-to-end detection without anchors
- Hungarian matching for training

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Lê Nguyễn Thành Tài

## Acknowledgments

- RetinaNet: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- Deformable DETR: [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)
