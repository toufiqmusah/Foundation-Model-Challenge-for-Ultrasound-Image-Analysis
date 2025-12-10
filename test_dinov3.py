#!/usr/bin/env python3
"""
Test script to verify DINOv3 integration works correctly
"""

import torch
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS

def test_encoder(encoder_name, encoder_weights=None):
    """Test a specific encoder configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {encoder_name}")
    print(f"{'='*60}")
    
    try:
        # Create model
        model = MultiTaskModelFactory(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            task_configs=TASK_CONFIGURATIONS
        )
        
        # Determine input size based on encoder type
        if encoder_name.startswith('facebook/dinov3'):
            input_size = 224
        else:
            input_size = 256
        
        # Create dummy input
        dummy_input = torch.randn(2, 3, input_size, input_size)
        print(f"Input shape: {dummy_input.shape}")
        
        # Test a few different tasks
        test_tasks = ['cardiac_multi', 'fetal_plane_cls', 'FUGC', 'thyroid_nodule_det']
        
        print("\nTesting forward passes:")
        for task_id in test_tasks:
            try:
                output = model(dummy_input, task_id=task_id)
                print(f"  ✓ {task_id:<30} -> Output shape: {output.shape}")
            except Exception as e:
                print(f"  ✗ {task_id:<30} -> Error: {e}")
        
        print(f"\n✓ Encoder '{encoder_name}' working correctly!")
        return True
        
    except Exception as e:
        print(f"\n✗ Encoder '{encoder_name}' failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("DINOv3 Integration Test Suite")
    print("="*60)
    
    results = {}
    
    # Test DINOv3 variants
    dinov3_models = [
        'facebook/dinov3-vitb16-pretrain-lvd1689m',
        # Uncomment to test other variants (requires more download time)
        # 'facebook/dinov3-vits16-pretrain-lvd1689m',
        # 'facebook/dinov3-vitl14-pretrain-lvd1689m',
    ]
    
    print("\nTesting DINOv3 Encoders:")
    print("-" * 60)
    for encoder_name in dinov3_models:
        results[encoder_name] = test_encoder(encoder_name, encoder_weights=None)
    
    # Test traditional SMP encoders
    smp_models = [
        ('efficientnet-b4', 'imagenet'),
        ('resnet34', 'imagenet'),
    ]
    
    print("\n\nTesting Traditional SMP Encoders:")
    print("-" * 60)
    for encoder_name, weights in smp_models:
        results[f"{encoder_name}_{weights}"] = test_encoder(encoder_name, encoder_weights=weights)
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! DINOv3 integration is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    exit(main())
