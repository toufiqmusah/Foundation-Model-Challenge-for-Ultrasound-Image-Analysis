#!/usr/bin/env python3
"""
Example script showing how to switch between different encoders
"""

import torch
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS

def compare_encoders():
    """Compare different encoder architectures"""
    
    # Example configurations
    encoders = {
        'DINOv3 ViT-B/16': {
            'encoder_name': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'encoder_weights': None,
            'input_size': 224
        },
        'DINOv3 ViT-S/16': {
            'encoder_name': 'facebook/dinov3-vits16-pretrain-lvd1689m',
            'encoder_weights': None,
            'input_size': 224
        },
        'EfficientNet-B4': {
            'encoder_name': 'efficientnet-b4',
            'encoder_weights': 'imagenet',
            'input_size': 256
        },
        'ResNet-34': {
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet',
            'input_size': 256
        }
    }
    
    print("="*70)
    print("Encoder Comparison")
    print("="*70)
    
    for name, config in encoders.items():
        print(f"\n{name}")
        print("-" * 70)
        print(f"  Model: {config['encoder_name']}")
        print(f"  Input Size: {config['input_size']}x{config['input_size']}")
        
        try:
            # Create model
            model = MultiTaskModelFactory(
                encoder_name=config['encoder_name'],
                encoder_weights=config['encoder_weights'],
                task_configs=TASK_CONFIGURATIONS
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"  Total Parameters: {total_params:,}")
            print(f"  Trainable Parameters: {trainable_params:,}")
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, config['input_size'], config['input_size'])
            with torch.no_grad():
                output = model(dummy_input, task_id='cardiac_multi')
            
            print(f"  Output Shape: {output.shape}")
            print(f"  Status: ✓ Working")
            
        except Exception as e:
            print(f"  Status: ✗ Error - {str(e)[:50]}")

if __name__ == '__main__':
    # Uncomment the line below to run the comparison
    # Note: First run will download models from HuggingFace
    compare_encoders()
    
    print("\n" + "="*70)
    print("Quick Usage Examples")
    print("="*70)
    
    print("\n1. Train with DINOv3 ViT-B/16:")
    print("   python train.py --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m")
    
    print("\n2. Train with DINOv3 ViT-S/16 (memory efficient):")
    print("   python train.py --encoder_name facebook/dinov3-vits16-pretrain-lvd1689m")
    
    print("\n3. Train with EfficientNet-B4 (original):")
    print("   python train.py --encoder_name efficientnet-b4 --encoder_weights imagenet")
    
    print("\n4. Inference with DINOv3:")
    print("   python model.py --encoder_name facebook/dinov3-vitb16-pretrain-lvd1689m \\")
    print("                   --model_path best_model.pth")
    
    print("\n" + "="*70)
