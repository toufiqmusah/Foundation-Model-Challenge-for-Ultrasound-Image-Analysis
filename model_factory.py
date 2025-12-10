import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import List, Dict, Optional
from transformers import AutoModel, AutoImageProcessor

# Task configuration list
TASK_CONFIGURATIONS = [
    {'task_name': 'Regression', 'num_classes': 2, 'task_id': 'FUGC'},
    {'task_name': 'Regression', 'num_classes': 3, 'task_id': 'IUGC'},
    {'task_name': 'Regression', 'num_classes': 2, 'task_id': 'fetal_femur'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'breast_2cls'},
    {'task_name': 'classification', 'num_classes': 3, 'task_id': 'breast_3cls'},
    {'task_name': 'classification', 'num_classes': 8, 'task_id': 'fetal_head_pos_cls'},
    {'task_name': 'classification', 'num_classes': 6, 'task_id': 'fetal_plane_cls'},
    {'task_name': 'classification', 'num_classes': 8, 'task_id': 'fetal_sacral_pos_cls'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'liver_lesion_2cls'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'lung_2cls'},
    {'task_name': 'classification', 'num_classes': 3, 'task_id': 'lung_disease_3cls'},
    {'task_name': 'classification', 'num_classes': 6, 'task_id': 'organ_cls'},
    {'task_name': 'detection', 'num_classes': 1, 'task_id': 'spinal_cord_injury_loc'},
    {'task_name': 'detection', 'num_classes': 1, 'task_id': 'thyroid_nodule_det'},
    {'task_name': 'detection', 'num_classes': 1, 'task_id': 'uterine_fibroid_det'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'breast_lesion'},
    {'task_name': 'segmentation', 'num_classes': 4, 'task_id': 'cardiac_multi'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'carotid_artery'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'cervix'},
    {'task_name': 'segmentation', 'num_classes': 3, 'task_id': 'cervix_multi'},
    {'task_name': 'segmentation', 'num_classes': 5, 'task_id': 'fetal_abdomen_multi'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'fetal_head'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'fetal_heart'},
    {'task_name': 'segmentation', 'num_classes': 3, 'task_id': 'head_symphysis_multi'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'lung'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'ovary_tumor'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'thyroid_nodule'},
]

# ====================================================================
# --- 1. DINOv3 Encoder Wrapper ---
# ====================================================================

class DINOv3Encoder(nn.Module):
    """Wrapper for DINOv3 to match SMP encoder interface."""
    def __init__(self, model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"):
        super().__init__()
        print(f"Loading DINOv3 model: {model_name}")
        self.dinov3 = AutoModel.from_pretrained(model_name)
        
        # DINOv3 ViT-B/16 architecture details
        # Input: 224x224 (or flexible with interpolation)
        # Patch size: 16x16
        # Hidden size: 768 (for ViT-B)
        self.hidden_size = self.dinov3.config.hidden_size
        
        # For SMP compatibility, we need to define out_channels as a list
        # Simulating multi-scale features by extracting from different layers
        # DINOv3 has 12 transformer blocks for ViT-B
        self.out_channels = [
            3,                    # Original input (stage 0)
            self.hidden_size,     # Early layers (stage 1)
            self.hidden_size,     # Mid layers (stage 2) 
            self.hidden_size,     # Late layers (stage 3)
            self.hidden_size      # Final layer (stage 4)
        ]
        
        # Layer indices to extract features from (0-indexed, 11 = last layer)
        self.feature_layers = [2, 5, 8, 11]  # Extract from layers 3, 6, 9, 12
        
    def forward(self, x):
        """
        Extract multi-scale features from DINOv3.
        Returns list of feature maps compatible with SMP decoders.
        """
        batch_size = x.shape[0]
        
        # Get hidden states from all layers
        outputs = self.dinov3(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (B, N+1, hidden_size)
        
        features = [x]  # Stage 0: original input
        
        # Extract features from selected layers
        for layer_idx in self.feature_layers:
            hidden = hidden_states[layer_idx]  # (B, N+1, hidden_size)
            
            # Remove CLS token
            patch_tokens = hidden[:, 1:, :]  # (B, N, hidden_size)
            
            # Reshape to spatial dimensions
            # Assuming input is square and patches are 16x16
            h = w = int(patch_tokens.shape[1] ** 0.5)
            spatial_features = patch_tokens.transpose(1, 2).reshape(
                batch_size, self.hidden_size, h, w
            )
            features.append(spatial_features)
        
        return features

# Task specific heads

class SmpClassificationHead(nn.Module):
    """Wrapper for SMP Classification Head."""
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = smp.base.ClassificationHead(
            in_channels=in_channels,
            classes=num_classes,
            pooling="avg",
            dropout=0.2,
            activation=None,
        )
        
    def forward(self, features: list):
        # Use the last feature map from encoder
        return self.head(features[-1])

class RegressionHead(nn.Module):
    """Custom head for regression tasks."""
    def __init__(self, in_channels: int, num_points: int):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        # Output dimension is num_points * 2 (x, y)
        self.linear = nn.Linear(in_channels, num_points * 2)

    def forward(self, features: list):
        x = self.pooling(features[-1])
        x = self.flatten(x)
        return self.linear(x)

class FPNGridDetectionHead(nn.Module):
    """Detection head designed for FPN outputs."""
    def __init__(self, fpn_out_channels: int, num_classes: int = 1, num_anchors: int = 1):
        super().__init__()
        mid_channels = 128
        num_outputs = num_anchors * (4 + num_classes)
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(fpn_out_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, num_outputs, kernel_size=1)
        )

    def forward(self, fpn_features: torch.Tensor):
        # Input fpn_features is already a single fused tensor from FPN Decoder
        predictions_map = self.conv_block(fpn_features)
        
        # Apply sigmoid to bbox coordinates (first 4 channels)
        predictions_map[:, :4] = torch.sigmoid(predictions_map[:, :4])
        
        return predictions_map

# ====================================================================
# --- 2. Multi-Task Model Factory ---
# ====================================================================

class MultiTaskModelFactory(nn.Module):
    def __init__(self, encoder_name: str, encoder_weights: Optional[str], task_configs: List[Dict]):
        super().__init__()
        
        # Check if using DINOv3 or traditional SMP encoder
        self.use_dinov3 = encoder_name.startswith('facebook/dinov3')
        
        if self.use_dinov3:
            # Initialize DINOv3 encoder
            print(f"Initializing DINOv3 encoder: {encoder_name}")
            self.encoder = DINOv3Encoder(model_name=encoder_name)
            
            # Create custom FPN decoder for DINOv3
            self.fpn_decoder = smp.fpn.decoder.FPNDecoder(
                encoder_channels=self.encoder.out_channels,
                encoder_depth=5,
                pyramid_channels=256,
                segmentation_channels=128,
                dropout=0.2,
                merge_policy='add'
            )
        else:
            # Initialize shared SMP encoder (EfficientNet, ResNet, etc.)
            print(f"Initializing SMP encoder: {encoder_name}")
            self.encoder = smp.encoders.get_encoder(
                name=encoder_name,
                in_channels=3,
                depth=5,
                weights=encoder_weights,
            )
            
            # Initialize shared FPN decoder
            temp_fpn_model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1, 
            )
            self.fpn_decoder = temp_fpn_model.decoder
        
        # Initialize task heads
        self.heads = nn.ModuleDict()
        
        print(f"Creating heads for {len(task_configs)} tasks...")
        for config in task_configs:
            task_id = config['task_id']
            task_name = config['task_name']
            num_classes = config['num_classes']
            
            head_module = None
            if task_name == 'segmentation':
                head_module = smp.base.SegmentationHead(
                    in_channels=self.fpn_decoder.out_channels, 
                    out_channels=num_classes, 
                    kernel_size=1,
                    upsampling=4 
                )

            elif task_name == 'classification':
                head_module = SmpClassificationHead(
                    in_channels=self.encoder.out_channels[-1],
                    num_classes=num_classes
                )

            elif task_name == 'Regression':
                num_points = config['num_classes']
                head_module = RegressionHead(
                    in_channels=self.encoder.out_channels[-1],
                    num_points=num_points
                )

            elif task_name == 'detection':
                head_module = FPNGridDetectionHead(
                    fpn_out_channels=self.fpn_decoder.out_channels,
                    num_classes=num_classes
                )

            if head_module:
                self.heads[task_id] = head_module
            else:
                print(f"Warning: Unknown task type '{task_name}' for {task_id}")

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        features = self.encoder(x)
        
        if task_id not in self.heads:
            raise ValueError(f"Task ID '{task_id}' not found.")

        task_config = next((item for item in TASK_CONFIGURATIONS if item["task_id"] == task_id), None)
        task_name = task_config['task_name'] if task_config else None

        # Route features based on task type
        if task_name in ['segmentation', 'detection']:
            # Use FPN features for dense prediction tasks
            fpn_features = self.fpn_decoder(features)
            output = self.heads[task_id](fpn_features)
        else: 
            # Use encoder features directly for global prediction tasks
            output = self.heads[task_id](features)
            
        return output

# Example usage

if __name__ == '__main__':
    model = MultiTaskModelFactory(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        task_configs=TASK_CONFIGURATIONS
    )

    print("\n--- Forward Pass Test ---")
    dummy_image_batch = torch.randn(2, 3, 256, 256) # Reduced batch size for test

    # Test specific tasks
    test_tasks = ['cardiac_multi', 'fetal_plane_cls', 'FUGC', 'thyroid_nodule_det']
    
    for t_id in test_tasks:
        try:
            out = model(dummy_image_batch, task_id=t_id)
            print(f"Task: {t_id:<25} | Output Shape: {out.shape}")
        except Exception as e:
            print(f"Task: {t_id:<25} | Error: {e}")
