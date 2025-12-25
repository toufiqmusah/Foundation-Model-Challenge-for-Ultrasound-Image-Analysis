import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch.losses as smp_losses
import numpy as np
import random
import argparse
from transformers import AutoImageProcessor

# Import local modules
from dataset import MultiTaskDataset, MultiTaskUniformSampler
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS
from utils import (
    multi_task_collate_fn, 
    evaluate, 
    DetectionLoss, 
    set_seed
)

# Training configuration
LEARNING_RATE = 1e-4
BATCH_SIZE = 20
NUM_EPOCHS = 50 
DATA_ROOT_PATH = '/root/baseline/train'
ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
RANDOM_SEED = 42
MODEL_SAVE_PATH = 'best_model.pth' 
VAL_SPLIT = 0.1

def main(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, data_root_path=DATA_ROOT_PATH, 
         encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, pretrain_weight_path=None):
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")
    print(f"Encoder: {encoder_name}")
    print(f"Encoder weights: {encoder_weights}")
    if pretrain_weight_path:
        print(f"Pretrain checkpoint: {pretrain_weight_path}")
    
    # Check if using DINOv3
    use_dinov3 = encoder_name.startswith('facebook/dinov3')

    # Data loading and splitting
    # Training transforms with augmentation
    if use_dinov3:
        print("Using DINOv3 preprocessing pipeline")
        # DINOv3 uses 224x224 by default
        train_transforms = A.Compose([
            A.Resize(224, 224), 
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))
        
        val_transforms = A.Compose([
            A.Resize(224, 224),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))
        
        # Load DINOv3 processor for normalization
        processor = AutoImageProcessor.from_pretrained(encoder_name)
    else:
        print("Using standard preprocessing pipeline")
        train_transforms = A.Compose([
            A.Resize(256, 256), 
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1), 
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))
        
        val_transforms = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))
        
        processor = None

    # Create full dataset to get indices
    temp_dataset = MultiTaskDataset(data_root=data_root_path, transforms=train_transforms, processor=processor)
    dataset_size = len(temp_dataset)
    val_size = int(dataset_size * VAL_SPLIT)
    train_size = dataset_size - val_size
    
    # Split indices
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = list(range(dataset_size))
    train_indices, val_indices = torch.utils.data.random_split(indices, [train_size, val_size], generator=generator)
    
    # Create separate datasets with different transforms
    train_dataset = MultiTaskDataset(data_root=data_root_path, transforms=train_transforms, processor=processor)
    val_dataset = MultiTaskDataset(data_root=data_root_path, transforms=val_transforms, processor=processor)
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Fix dataframe reference for subset
    train_subset.dataframe = train_dataset.dataframe.iloc[train_indices.indices].reset_index(drop=True)
    
    train_sampler = MultiTaskUniformSampler(train_subset, batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_subset, 
        batch_sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, 
        batch_size=8,
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    # Model and loss setup
    model = MultiTaskModelFactory(
        encoder_name=encoder_name, 
        encoder_weights=encoder_weights if not use_dinov3 else None, 
        task_configs=TASK_CONFIGURATIONS
    ).to(device)
    
    # Load pretrained weights if provided
    if pretrain_weight_path:
        print(f"\n--- Loading pretrained weights from {pretrain_weight_path} ---")
        try:
            checkpoint = torch.load(pretrain_weight_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print("✓ Successfully loaded pretrained weights for continued training")
        except Exception as e:
            print(f"⚠ Warning: Could not load pretrained weights: {e}")
            print("Continuing with randomly initialized weights...")
    
    loss_functions = {
        'segmentation': smp_losses.DiceLoss(mode='multiclass'), 
        'classification': nn.CrossEntropyLoss(),
        'Regression': nn.MSELoss(), 
        'detection': DetectionLoss()
    }
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}

    # Optimization setup
    print("\n--- Setting parameter groups ---")
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': LEARNING_RATE * 1},
    ]
    print(f"  - Shared Encoder                 -> LR: {LEARNING_RATE * 1}")
    
    for task_id, head in model.heads.items():
        lr_multiplier = 10.0
        current_lr = LEARNING_RATE * lr_multiplier
        param_groups.append({'params': head.parameters(), 'lr': current_lr})
        print(f"  - Task Head '{task_id:<25}' -> LR: {current_lr}")

    optimizer = optim.AdamW(param_groups)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    print("\n--- Cosine Annealing Scheduler configured ---")

    best_val_score = -float('inf')
    print("\n" + "="*50 + "\n--- Start Training ---")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in loop:
            images = batch['image'].to(device)
            task_ids = batch['task_id']
            # Manually stack labels list to tensor
            labels = torch.stack(batch['label']).to(device)

            # All samples in batch belong to the same task due to sampler
            current_task_id = task_ids[0]
            task_name = task_id_to_name[current_task_id]

            outputs = model(images, task_id=current_task_id)
            
            # Grid-based detection logic
            if task_name == 'detection':
                _, _, h, w = outputs.shape
                
                # Calculate center of GT box (normalized)
                gt_center_x = (labels[:, 0] + labels[:, 2]) / 2.0
                gt_center_y = (labels[:, 1] + labels[:, 3]) / 2.0

                # Map to grid coordinates
                coord_h = torch.clamp((gt_center_y * h).long(), 0, h - 1)
                coord_w = torch.clamp((gt_center_x * w).long(), 0, w - 1)

                # Extract prediction from the specific grid cell
                final_outputs = torch.zeros((images.shape[0], 5), device=device)
                for i in range(images.shape[0]):
                    final_outputs[i] = outputs[i, :, coord_h[i], coord_w[i]]
            else:
                final_outputs = outputs
            
            loss = loss_functions[task_name](final_outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_losses[current_task_id].append(loss.item())
            loop.set_postfix(loss=loss.item(), task=current_task_id, lr=scheduler.get_last_lr()[0])

        # Train reporting
        print("\n--- Epoch {} Average Train Loss Report ---".format(epoch + 1))
        sorted_task_ids = sorted(epoch_train_losses.keys())
        for task_id in sorted_task_ids:
            avg_loss = np.mean(epoch_train_losses[task_id])
            print(f"  - Task '{task_id:<25}': {avg_loss:.4f}")
        print("-" * 40)

        # Validation
        val_results_df = evaluate(model, val_loader, device)
        
        score_cols = [col for col in val_results_df.columns if 'MAE' not in col and isinstance(val_results_df[col].iloc[0], (int, float))]
        avg_val_score = 0
        if not val_results_df.empty and score_cols:
            avg_val_score = val_results_df[score_cols].mean().mean()

        print("\n--- Epoch {} Validation Report ---".format(epoch + 1))
        if not val_results_df.empty:
            print(val_results_df.to_string(index=False))
        print(f"--- Average Val Score (Higher is better): {avg_val_score:.4f} ---")

        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> New best model saved! Score improved to: {best_val_score:.4f}\n")
        
        # Update scheduler
        scheduler.step()

    print(f"\n--- Training Finished ---\nBest model saved at: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multi-task ultrasound model')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, 
                        help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of training epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT_PATH,
                        help=f'Root directory for training data (default: {DATA_ROOT_PATH})')
    parser.add_argument('--encoder_name', type=str, default=ENCODER,
                        help=f'Encoder backbone name (default: {ENCODER}). Examples: facebook/dinov3-vitb16-pretrain-lvd1689m')
    parser.add_argument('--encoder_weights', type=str, default=ENCODER_WEIGHTS,
                        help=f'Encoder weights (default: {ENCODER_WEIGHTS}). Ignored for DINOv3 models.')
    parser.add_argument('--pretrain_weights', type=str, default=None,
                        help='Path to pretrained model checkpoint for continued finetuning (default: None)')
    
    args = parser.parse_args()
    
    main(
        batch_size=args.batch_size, 
        num_epochs=args.num_epochs, 
        data_root_path=args.data_root,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        pretrain_weight_path=args.pretrain_weights
    )