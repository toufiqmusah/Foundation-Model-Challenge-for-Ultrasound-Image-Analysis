# Quick Start: DINOv3 Integration

## Installation

```bash
pip install -r requirements.txt
```

## Training

### DINOv3 ViT-Base/16 (Recommended)
```bash
python train.py \
    --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --batch_size 16 \
    --num_epochs 50 \
    --data_root /root/baseline/train
```

### DINOv3 ViT-Small/16 (Memory Efficient)
```bash
python train.py \
    --encoder_name facebook/dinov3-vits16-pretrain-lvd1689m \
    --batch_size 20 \
    --num_epochs 50 \
    --data_root /root/baseline/train
```

### DINOv3 ViT-Large/14 (Best Performance)
```bash
python train.py \
    --encoder_name facebook/dinov3-vitl14-pretrain-lvd1689m \
    --batch_size 8 \
    --num_epochs 50 \
    --data_root /root/baseline/train
```

### Traditional EfficientNet-B4 (Original)
```bash
python train.py \
    --encoder_name efficientnet-b4 \
    --encoder_weights imagenet \
    --batch_size 20 \
    --num_epochs 50 \
    --data_root /root/baseline/train
```

## Inference

### With DINOv3
```bash
python model.py \
    --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --data_root /root/baseline/test \
    --output_dir predictions/ \
    --model_path best_model.pth \
    --batch_size 8
```

### With EfficientNet
```bash
python model.py \
    --encoder_name efficientnet-b4 \
    --data_root /root/baseline/test \
    --output_dir predictions/ \
    --model_path best_model.pth \
    --batch_size 8
```

## Testing

Test the integration:
```bash
python test_dinov3.py
```

## Available DINOv3 Variants

| Model | HuggingFace ID | Memory | Speed | Quality |
|-------|---------------|--------|-------|---------|
| ViT-S/14 | `facebook/dinov3-vits14-pretrain-lvd1689m` | ~2GB | Fast | Good |
| ViT-S/16 | `facebook/dinov3-vits16-pretrain-lvd1689m` | ~2GB | Fast | Good |
| **ViT-B/14** | `facebook/dinov3-vitb14-pretrain-lvd1689m` | ~4GB | Medium | Better |
| **ViT-B/16** ⭐ | `facebook/dinov3-vitb16-pretrain-lvd1689m` | ~4GB | Medium | Better |
| ViT-L/14 | `facebook/dinov3-vitl14-pretrain-lvd1689m` | ~8GB | Slow | Best |
| ViT-L/16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | ~8GB | Slow | Best |
| ViT-G/14 | `facebook/dinov3-vitg14-pretrain-lvd1689m` | ~16GB | Very Slow | Excellent |

⭐ Recommended starting point

## Key Points

1. **Same encoder for train & inference**: Use the same `--encoder_name` for both!
2. **Input size**: DINOv3 uses 224x224 (handled automatically)
3. **Weights param**: Not used for DINOv3 (handled automatically)
4. **Memory**: Reduce `--batch_size` if OOM errors occur
5. **First run**: Downloads model from HuggingFace (~300MB-1GB per model)
