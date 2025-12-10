# Quick Start: Using DINOv3 Backbone

## What Changed?

The codebase now supports **DINOv3 Vision Transformers** as backbone encoders, in addition to the existing EfficientNet/ResNet options.

## How to Use

### 1. Training with DINOv3

```bash
# DINOv3 ViT-Base (recommended)
python train.py --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m

# DINOv3 ViT-Small (faster, less memory)
python train.py --encoder_name facebook/dinov3-vits16-pretrain-lvd1689m

# DINOv3 ViT-Large (best performance)
python train.py --encoder_name facebook/dinov3-vitl16-pretrain-lvd1689m
```

### 2. Inference with DINOv3

```bash
python model.py \
    --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --data_root /path/to/data \
    --output_dir predictions/
```

### 3. Still Using EfficientNet (default)

```bash
# Default behavior unchanged
python train.py --encoder_name efficientnet-b4 --encoder_weights imagenet
```

## Available DINOv3 Models

| Model | Parameters | Best For |
|-------|-----------|----------|
| `facebook/dinov3-vits16-pretrain-lvd1689m` | 22M | Fast inference, limited GPU |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | 86M | **Recommended starting point** |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | 304M | Maximum performance |
| `facebook/dinov3-vitg16-pretrain-lvd1689m` | 1.1B | State-of-the-art (needs A100) |

## Installation

```bash
pip install transformers>=4.30.0 accelerate
```

Or:

```bash
pip install -r requirements.txt
```

## Testing

Verify the integration works:

```bash
python test_dinov3.py
```

## Key Points

✅ **Seamless switching** - Just change `--encoder_name` parameter  
✅ **All tasks supported** - Segmentation, classification, detection, regression  
✅ **Automatic preprocessing** - No manual config needed  
✅ **Backward compatible** - Existing EfficientNet/ResNet code still works  

## Example Commands

```bash
# Train with DINOv3 on custom data
python train.py \
    --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --data_root /path/to/train \
    --batch_size 16 \
    --num_epochs 50

# Run inference
python model.py \
    --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --data_root /path/to/test \
    --output_dir predictions/ \
    --batch_size 8
```

## Need More Details?

See [DINOV3_GUIDE.md](DINOV3_GUIDE.md) for comprehensive documentation.
