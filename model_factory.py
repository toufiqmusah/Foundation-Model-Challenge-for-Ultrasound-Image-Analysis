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
        
        # Get the hidden size from config
        self.hidden_size = self.dinov3.config.hidden_size  # 768 for ViT-B
        
        # For SMP compatibility, define out_channels
        # We'll use the pooled output (global representation) as the main feature
        self.out_channels = [
            3,                    # Original input (for reference)
            self.hidden_size,     # Pooled output feature
            self.hidden_size,
            self.hidden_size,
            self.hidden_size
        ]
        
    def forward(self, x):
        """
        Extract features from DINOv3.
        Returns list compatible with SMP encoder interface.
        """
        # Forward through DINOv3
        outputs = self.dinov3(pixel_values=x)
        
        # Get pooled output (CLS token representation)
        pooled = outputs.pooler_output  # Shape: (B, hidden_size)
        
        # For compatibility with dense prediction tasks, we need spatial features
        # We'll get the last hidden state and reshape it to spatial dimensions
        last_hidden = outputs.last_hidden_state  # (B, N+1, hidden_size)
        
        # Remove CLS token and reshape to spatial grid
        batch_size = x.shape[0]
        patch_tokens = last_hidden[:, 1:, :]  # (B, N, hidden_size)
        
        # Calculate spatial dimensions (assuming 16x16 patches)
        h = x.shape[2] // 16
        w = x.shape[3] // 16
        
        spatial_features = patch_tokens.transpose(1, 2).reshape(
            batch_size, self.hidden_size, h, w
        )
        
        # Return features at multiple "scales" (same resolution but from different perspectives)
        return [
            x,                   # Original input
            spatial_features,    # Spatial patch features
            spatial_features,    # (repeat for multi-scale compatibility)
            spatial_features,
            spatial_features
        ]

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
            
            # DINOv3 outputs uniform channel dimensions (all 768 for ViT-B)
            # But FPN decoder expects varying channel dimensions like traditional CNNs
            # We'll create adapter convolutions to convert DINOv3 features to expected dimensions
            
            # Expected channel dimensions for FPN (mimicking ResNet-34 structure)
            target_channels = [3, 64, 128, 256, 512]  # Matching typical CNN encoder
            
            # Create 1x1 convolutions to adapt DINOv3's uniform channels to target channels
            self.channel_adapters = nn.ModuleList()
            for i, (din_ch, target_ch) in enumerate(zip(self.encoder.out_channels, target_channels)):
                if i == 0:
                    # First stage is the input image, no adaptation needed
                    self.channel_adapters.append(nn.Identity())
                else:
                    # Adapt from DINOv3's hidden_size to target channels
                    self.channel_adapters.append(
                        nn.Conv2d(din_ch, target_ch, kernel_size=1, bias=False)
                    )
            
            # Now create FPN decoder with the target channel dimensions
            dummy_fpn = smp.FPN(
                encoder_name='resnet34',  # Similar channel structure
                encoder_weights=None,
                in_channels=3,
                classes=1
            )
            self.fpn_decoder = dummy_fpn.decoder
            
            # Store the adapted channel dimensions for head creation
            self.adapted_channels = target_channels
            
        else:
            # Initialize shared SMP encoder (EfficientNet, ResNet, etc.)
            print(f"Initializing SMP encoder: {encoder_name}")
         # Initialize shared SMP encoder (EfficientNet, ResNet, etc.)
            # print(f"Initializing SMP encoder: {encoder_name}")
            # self.encoder = smp.encoders.get_encoder(
            #     name=encoder_name,
            #     in_channels=3,
            #     depth=5,
            #     weights=encoder_weights,
            # )

            checkpoint = "smp-hub/segformer-b4-512x512-ade-160k"
            self.encoder = smp.from_pretrained(checkpoint)
            
            # Initialize shared FPN decoder
            temp_fpn_model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1, 
            )
            self.fpn_decoder = temp_fpn_model.decoder
            self.adapted_channels = None  # No adaptation needed for standard encoders
        
        # Determine the channel dimensions to use for head creation
        if self.use_dinov3:
            feature_channels = self.adapted_channels
        else:
            feature_channels = self.encoder.out_channels
        
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
        
        # Apply channel adapters if using DINOv3
        if self.use_dinov3:
            adapted_features = []
            for i, (feat, adapter) in enumerate(zip(features, self.channel_adapters)):
                adapted_features.append(adapter(feat))
            features = adapted_features
        
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
