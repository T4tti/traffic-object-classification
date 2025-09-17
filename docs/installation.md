# Installation Guide

This guide covers the installation and setup process for the Traffic Object Classification project.

## System Requirements

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA GPU with 6GB VRAM (GTX 1060, RTX 2060, or equivalent)
- CPU: 4 cores, 2.0 GHz
- RAM: 16GB
- Storage: 50GB available space

**Recommended:**
- GPU: NVIDIA GPU with 11GB+ VRAM (RTX 2080 Ti, RTX 3080, or better)
- CPU: 8+ cores, 3.0 GHz
- RAM: 32GB
- Storage: 100GB SSD

### Software Requirements

- Python 3.8 or higher
- CUDA 11.0 or higher (for GPU support)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/T4tti/traffic-object-classification.git
cd traffic-object-classification
```

### 2. Create Virtual Environment

#### Using conda (recommended):

```bash
conda create -n traffic-detection python=3.8
conda activate traffic-detection
```

#### Using venv:

```bash
python -m venv traffic-detection
source traffic-detection/bin/activate  # Linux/Mac
# or
traffic-detection\Scripts\activate     # Windows
```

### 3. Install Dependencies

#### Basic Installation:

```bash
pip install -r requirements.txt
```

#### Development Installation:

```bash
pip install -r requirements.txt
pip install -e .
```

#### GPU Support:

Ensure CUDA is properly installed and compatible with PyTorch:

```bash
# Check CUDA version
nvcc --version

# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 5. Install Pre-commit Hooks (Optional)

For development:

```bash
pre-commit install
```

## Package Dependencies

### Core Dependencies

- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **pytorch-lightning**: Training framework
- **transformers**: Transformer models (for DETR)

### Computer Vision

- **opencv-python**: Image processing
- **Pillow**: Image handling
- **albumentations**: Data augmentation

### Data Handling

- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scipy**: Scientific computing

### Configuration

- **PyYAML**: YAML parsing
- **omegaconf**: Configuration management
- **hydra-core**: Configuration framework

### Visualization

- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **wandb**: Experiment tracking
- **tensorboard**: Training monitoring

### Object Detection Specific

- **pycocotools**: COCO dataset utilities
- **detectron2**: Reference implementations

### Development Tools

- **pytest**: Testing framework
- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **pre-commit**: Git hooks

## Docker Installation (Alternative)

### Build Docker Image

```bash
docker build -t traffic-detection .
```

### Run Container

```bash
# With GPU support
docker run --gpus all -it -v $(pwd):/workspace traffic-detection

# CPU only
docker run -it -v $(pwd):/workspace traffic-detection
```

## Troubleshooting

### Common Installation Issues

#### 1. CUDA Compatibility Issues

**Error**: `RuntimeError: CUDA error: no kernel image is available for execution on the device`

**Solution**: Install PyTorch with the correct CUDA version:

```bash
# Check your CUDA version
nvidia-smi

# Install compatible PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Memory Issues During Installation

**Error**: `MemoryError` during pip install

**Solution**: Use pip with no cache:

```bash
pip install --no-cache-dir -r requirements.txt
```

#### 3. Detectron2 Installation Issues

**Error**: Detectron2 compilation fails

**Solution**: Install pre-built binaries:

```bash
# For CUDA 11.8
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch1.13/index.html
```

#### 4. PyCocoTools Issues on Windows

**Error**: PyCocoTools installation fails on Windows

**Solution**: Install Visual Studio Build Tools or use conda:

```bash
conda install pycocotools -c conda-forge
```

### Environment Verification

Run the verification script to check your installation:

```bash
python scripts/verify_installation.py
```

This script will check:
- Python version
- PyTorch installation and CUDA support
- Required packages
- Dataset accessibility
- Model loading capabilities

## Configuration

### Environment Variables

Set these environment variables for optimal performance:

```bash
# Linux/Mac
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Windows
set CUDA_VISIBLE_DEVICES=0
set OMP_NUM_THREADS=4
```

### Memory Management

For large datasets, configure these settings:

```bash
# Increase shared memory for data loading
ulimit -n 65536  # Linux/Mac
```

## Next Steps

After installation:

1. **Download Data**: Use the data download script
2. **Verify Setup**: Run verification scripts
3. **Explore Notebooks**: Start with data exploration notebook
4. **Run Training**: Train your first model

## Support

If you encounter installation issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/T4tti/traffic-object-classification/issues)
3. Create a new issue with your error details and system information

Include this information when reporting issues:

```bash
# System information
python --version
pip list | grep torch
nvidia-smi  # If using GPU
uname -a    # Linux/Mac
```