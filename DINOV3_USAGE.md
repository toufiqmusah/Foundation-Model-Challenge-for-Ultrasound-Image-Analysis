# DINOv3 Backbone Integration Guide

This guide explains how to use DINOv3 backbones with the multi-task ultrasound model.

## Overview

The model now supports both traditional SMP encoders (EfficientNet, ResNet, etc.) and Facebook's DINOv3 vision transformers as backbones. DINOv3 models are self-supervised vision transformers that provide strong feature representations.

## Available DINOv3 Models

You can use any of the following DINOv3 variants from HuggingFace:

### Small Models
- `facebook/dinov3-vits14-pretrain-lvd1689m` (ViT-Small, patch 14x14)
- `facebook/dinov3-vits16-pretrain-lvd1689m` (ViT-Small, patch 16x16)

### Base Models (Recommended)
- `facebook/dinov3-vitb14-pretrain-lvd1689m` (ViT-Base, patch 14x14)
- `facebook/dinov3-vitb16-pretrain-lvd1689m` (ViT-Base, patch 16x16) ⭐

### Large Models
- `facebook/dinov3-vitl14-pretrain-lvd1689m` (ViT-Large, patch 14x14)
- `facebook/dinov3-vitl16-pretrain-lvd1689m` (ViT-Large, patch 16x16)

### Giant Models
- `facebook/dinov3-vitg14-pretrain-lvd1689m` (ViT-Giant, patch 14x14)

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- `transformers` - HuggingFace library for loading DINOv3
- `accelerate` - For efficient model loading
- `safetensors` - For loading model weights

## Training with DINOv3

### Basic Training Command

```bash
python train.py \
    --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --batch_size 16 \
    --num_epochs 50 \
    --data_root /path/to/data
```

### Switching Between DINOv3 Variants

**Using ViT-Small (Fastest, less memory):**
```bash
python train.py --encoder_name facebook/dinov3-vits16-pretrain-lvd1689m
```

**Using ViT-Base (Balanced performance):**
```bash
python train.py --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m
```

**Using ViT-Large (Best performance, more memory):**
```bash
python train.py --encoder_name facebook/dinov3-vitl14-pretrain-lvd1689m
```

### Using Traditional SMP Encoders (EfficientNet, ResNet)

You can still use the original encoders:

```bash
python train.py \
    --encoder_name efficientnet-b4 \
    --encoder_weights imagenet \
    --batch_size 20
```

## Inference with DINOv3

### Basic Inference Command

```bash
python model.py \
    --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --data_root /path/to/test/data \
    --output_dir predictions/ \
    --model_path best_model.pth
```

### Important Notes

1. **Consistent Encoder**: Make sure to use the **same encoder** for inference that was used during training!

2. **Model Path**: The `--model_path` should point to the checkpoint saved during training.

3. **Memory Requirements**:
   - ViT-Small: ~2GB VRAM
   - ViT-Base: ~4GB VRAM  
   - ViT-Large: ~8GB VRAM
   - ViT-Giant: ~16GB VRAM

## Key Differences from EfficientNet

| Aspect | EfficientNet | DINOv3 |
|--------|--------------|---------|
| **Architecture** | CNN-based | Transformer-based |
| **Input Size** | 256x256 (default) | 224x224 (default) |
| **Preprocessing** | Standard normalization | DINOv3 processor |
| **Patch Size** | N/A | 14x14 or 16x16 |
| **Multi-scale Features** | Native pyramid | Extracted from layers |

## Code Example

### Programmatic Usage

```python
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS
import torch

# Create model with DINOv3
model = MultiTaskModelFactory(
    encoder_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
    encoder_weights=None,  # DINOv3 doesn't use this parameter
    task_configs=TASK_CONFIGURATIONS
)

# Forward pass
dummy_input = torch.randn(2, 3, 224, 224)  # Note: 224x224 for DINOv3
output = model(dummy_input, task_id='cardiac_multi')
print(f"Output shape: {output.shape}")
```

### Switching Backbones Dynamically

```python
# DINOv3 backbone
dinov3_model = MultiTaskModelFactory(
    encoder_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
    encoder_weights=None,
    task_configs=TASK_CONFIGURATIONS
)

# EfficientNet backbone
efficientnet_model = MultiTaskModelFactory(
    encoder_name='efficientnet-b4',
    encoder_weights='imagenet',
    task_configs=TASK_CONFIGURATIONS
)
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Use a smaller model variant or reduce batch size:
```bash
python train.py \
    --encoder_name facebook/dinov3-vits16-pretrain-lvd1689m \
    --batch_size 8
```

### Issue: Slow Download

**Solution**: DINOv3 models are downloaded from HuggingFace Hub on first use. Set cache directory:
```bash
export HF_HOME=/path/to/cache
python train.py --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m
```

### Issue: Import Error

**Solution**: Make sure transformers is installed:
```bash
pip install transformers accelerate safetensors
```

## Performance Tips

1. **Batch Size**: DINOv3 typically requires smaller batch sizes than CNNs due to memory requirements
2. **Learning Rate**: Consider using a lower learning rate (1e-5) for the DINOv3 encoder if fine-tuning
3. **Mixed Precision**: Use mixed precision training to reduce memory usage
4. **Gradient Checkpointing**: Enable for very large models (ViT-Giant)

## References

- [DINOv3 Paper](https://arxiv.org/abs/2304.07193)
- [HuggingFace Model Card](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
- [Official DINOv3 Repository](https://github.com/facebookresearch/dinov2)
