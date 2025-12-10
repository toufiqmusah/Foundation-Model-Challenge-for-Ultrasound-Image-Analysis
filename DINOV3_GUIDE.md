# DINOv3 Integration Guide

This guide explains how to use DINOv3 backbones in the Foundation Model Challenge for Ultrasound Image Analysis.

## Available DINOv3 Variants

All DINOv3 models from Meta AI are now supported. Here are the main variants:

### DINOv3 ViT-Base (Recommended for starting)
```bash
facebook/dinov3-vitb16-pretrain-lvd1689m
```
- **Parameters**: ~86M
- **Patch size**: 16×16
- **Hidden size**: 768
- **Input**: 224×224
- **Best for**: Balanced performance and speed

### DINOv3 ViT-Large
```bash
facebook/dinov3-vitl16-pretrain-lvd1689m
```
- **Parameters**: ~304M
- **Patch size**: 16×16
- **Hidden size**: 1024
- **Input**: 224×224
- **Best for**: Maximum performance, requires more GPU memory

### DINOv3 ViT-Small
```bash
facebook/dinov3-vits16-pretrain-lvd1689m
```
- **Parameters**: ~22M
- **Patch size**: 16×16
- **Hidden size**: 384
- **Input**: 224×224
- **Best for**: Fast inference, limited GPU memory

### DINOv3 ViT-Giant
```bash
facebook/dinov3-vitg16-pretrain-lvd1689m
```
- **Parameters**: ~1.1B
- **Patch size**: 16×16
- **Hidden size**: 1536
- **Input**: 224×224
- **Best for**: State-of-the-art results, requires A100/H100

## Usage

### Training with DINOv3

```bash
python train.py \
    --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --batch_size 16 \
    --num_epochs 50 \
    --data_root /path/to/data
```

### Inference with DINOv3

```bash
python model.py \
    --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --data_root /path/to/data \
    --output_dir predictions/ \
    --batch_size 8
```

### Switching Between Encoders

You can easily switch between DINOv3 and traditional encoders:

**EfficientNet-B4 (default)**:
```bash
python train.py --encoder_name efficientnet-b4 --encoder_weights imagenet
```

**DINOv3 ViT-Base**:
```bash
python train.py --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m
```

**ResNet34**:
```bash
python train.py --encoder_name resnet34 --encoder_weights imagenet
```

## Key Differences

### DINOv3 vs EfficientNet

| Feature | DINOv3 | EfficientNet-B4 |
|---------|--------|-----------------|
| Architecture | Vision Transformer | CNN |
| Input Size | 224×224 | 256×256 |
| Preprocessing | BitImageProcessor | Standard normalization |
| Pre-training | Self-supervised (LVD-142M) | Supervised (ImageNet-1k) |
| Feature Type | Patch embeddings | Spatial feature maps |

### Image Preprocessing

**DINOv3**:
- Resizes to 224×224
- Uses BitImageProcessor for normalization
- No need to specify mean/std manually

**Traditional Encoders (EfficientNet, ResNet, etc.)**:
- Resizes to 256×256
- Uses ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Converts to tensor with ToTensorV2()

## Model Architecture

The integration automatically handles:
1. **Multi-scale feature extraction**: Extracts features from layers 3, 6, 9, and 12 of the transformer
2. **Spatial reshaping**: Converts patch tokens back to spatial feature maps
3. **FPN decoder**: Uses custom FPN decoder compatible with transformer features
4. **Task heads**: All existing task heads (segmentation, classification, detection, regression) work seamlessly

## Performance Tips

### GPU Memory Optimization

For large models (ViT-L, ViT-G), reduce batch size:
```bash
python train.py \
    --encoder_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --batch_size 8  # Reduced from 20
```

### Mixed Precision Training

Enable automatic mixed precision for faster training:
```python
# In train.py, wrap training loop with:
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(images, task_id=current_task_id)
    loss = loss_functions[task_name](final_outputs, labels)
```

### Gradient Checkpointing

For very large models, enable gradient checkpointing:
```python
# In model_factory.py DINOv3Encoder.__init__
self.dinov3.gradient_checkpointing_enable()
```

## Troubleshooting

### Issue: Out of memory
**Solution**: Reduce batch size or use a smaller variant (ViT-S or ViT-B)

### Issue: Slow training
**Solution**: Reduce num_workers in DataLoader or use mixed precision training

### Issue: Model not loading
**Solution**: Ensure `transformers` library is installed: `pip install transformers>=4.30.0`

## Citation

If you use DINOv3 in your research, please cite:

```bibtex
@article{oquab2023dinov3,
  title={DINOv3: Towards Self-supervised Vision Transformers Beyond the Patch-level},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

## Additional Resources

- [DINOv3 Paper](https://arxiv.org/abs/2304.07193)
- [HuggingFace Model Hub](https://huggingface.co/collections/facebook/dinov3-65f9e3c3d1be5a23bb98eed2)
- [Official GitHub](https://github.com/facebookresearch/dinov2)
