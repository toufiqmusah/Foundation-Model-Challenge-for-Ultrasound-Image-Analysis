import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
import cv2
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from albumentations.core.transforms_interface import ImageOnlyTransform
from model_factory import TASK_CONFIGURATIONS  # Needed for task name mapping

def brightness_preserving_bihistogram_equalization(image):
    """
    Brightness Preserving Bi-Histogram Equalization (BBHE).
    Enhances contrast while preserving mean brightness.
    Args:
        image: RGB image as numpy array (H, W, 3)
    Returns:
        Enhanced image with same shape and dtype
    """
    # Convert to LAB color space to process only luminance
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    
    # Calculate mean brightness
    mean_brightness = np.mean(l_channel)
    
    # Build histogram
    hist, bins = np.histogram(l_channel.flatten(), bins=256, range=(0, 255))
    
    # Split histogram at mean
    mean_idx = int(np.round(mean_brightness))
    
    # Calculate CDFs for lower and upper parts
    hist_lower = hist[:mean_idx + 1]
    hist_upper = hist[mean_idx + 1:]
    
    cdf_lower = hist_lower.cumsum()
    cdf_upper = hist_upper.cumsum()
    
    # Normalize CDFs
    cdf_lower_norm = cdf_lower / (cdf_lower[-1] + 1e-10)  # Avoid division by zero
    cdf_upper_norm = cdf_upper / (cdf_upper[-1] + 1e-10)
    
    # Create lookup tables
    lut = np.zeros(256, dtype=np.float32)
    
    # Map lower part to [0, mean_brightness]
    for i in range(mean_idx + 1):
        lut[i] = cdf_lower_norm[i] * mean_brightness
    
    # Map upper part to [mean_brightness, 255]
    for i in range(len(cdf_upper_norm)):
        lut[mean_idx + 1 + i] = mean_brightness + cdf_upper_norm[i] * (255 - mean_brightness)
    
    # Apply lookup table
    l_channel_eq = np.interp(l_channel.flatten(), np.arange(256), lut).reshape(l_channel.shape)
    
    # Clip and convert back
    lab[:, :, 0] = np.clip(l_channel_eq, 0, 255).astype(np.uint8)
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced


class BBHE(ImageOnlyTransform):
    """Albumentations-compatible Brightness Preserving Bi-Histogram Equalization."""
    
    def __init__(self, always_apply=False, p=1.0):
        super(BBHE, self).__init__(always_apply, p)
    
    def apply(self, img, **params):
        return brightness_preserving_bihistogram_equalization(img)
    
    def get_transform_init_args_names(self):
        return ()

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def multi_task_collate_fn(batch):
    """
    Custom collate function to handle different label shapes in multi-task learning.
    Images are stacked; labels and task_ids remain as lists.
    """
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    task_ids = [item['task_id'] for item in batch]
    
    # Stack images as they have consistent dimensions
    images = torch.stack(images, 0)
    
    return {'image': images, 'label': labels, 'task_id': task_ids}

class DetectionLoss(nn.Module):
    """A simplified loss function for object detection (Cls + Reg)."""
    def __init__(self, classification_weight=1.0, box_regression_weight=8.0):
        super().__init__()
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.box_regression_loss = nn.L1Loss()
        self.cls_w, self.box_w = classification_weight, box_regression_weight

    def forward(self, predictions, targets):
        # predictions: [B, 5], targets: [B, 4]
        pred_boxes, pred_scores = predictions[:, :4], predictions[:, 4].squeeze(-1)
        
        # Filter valid targets (dummy targets are usually -1)
        valid_indices = (targets[:, 0] >= 0).view(-1)
        if not valid_indices.any(): 
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        cls_loss = self.classification_loss(pred_scores[valid_indices], torch.ones_like(pred_scores)[valid_indices])
        box_loss = self.box_regression_loss(pred_boxes[valid_indices], targets[valid_indices])
        
        return self.cls_w * cls_loss + self.box_w * box_loss

# --- Metric Calculations ---

def calculate_accuracy(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return accuracy_score(y_true, y_pred)

def calculate_f1_score(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def calculate_dice_coefficient(y_true, y_pred_logits):
    y_pred_mask = torch.argmax(y_pred_logits, dim=1)
    num_classes = y_pred_logits.shape[1]
    y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2)
    y_pred_one_hot = F.one_hot(y_pred_mask, num_classes=num_classes).permute(0, 3, 1, 2)
    intersection = torch.sum(y_true_one_hot[:, 1:] * y_pred_one_hot[:, 1:])
    union = torch.sum(y_true_one_hot[:, 1:]) + torch.sum(y_pred_one_hot[:, 1:])
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.item()

def calculate_mae(y_true, y_pred, image_size=(256, 256)):
    h, w = image_size
    y_true_px = y_true.cpu().numpy().copy()
    y_pred_px = y_pred.cpu().numpy().copy()
    y_true_px[:, 0::2] *= w; y_true_px[:, 1::2] *= h
    y_pred_px[:, 0::2] *= w; y_pred_px[:, 1::2] *= h
    return np.mean(np.abs(y_true_px - y_pred_px))

def calculate_iou(y_true, y_pred):
    y_true = y_true.cpu().numpy(); y_pred = y_pred.cpu().numpy()
    batch_ious = []
    for i in range(y_true.shape[0]):
        box_true, box_pred = y_true[i], y_pred[i]
        xA = max(box_true[0], box_pred[0]); yA = max(box_true[1], box_pred[1])
        xB = min(box_true[2], box_pred[2]); yB = min(box_true[3], box_pred[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        box_true_area = (box_true[2] - box_true[0]) * (box_true[3] - box_true[1])
        box_pred_area = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
        union_area = box_true_area + box_pred_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        batch_ious.append(iou)
    return np.mean(batch_ious)

def evaluate(model, val_loader, device):
    """
    Evaluation loop supporting multi-task batches.
    """
    model.eval()
    task_metrics = defaultdict(lambda: defaultdict(list))
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}
    
    with torch.no_grad():
        loop = tqdm(val_loader, desc="[Validation]")
        for batch in loop:
            images = batch['image'].to(device)
            labels = batch['label']
            task_ids = batch['task_id']

            unique_tasks_in_batch = set(task_ids)

            for task_id in unique_tasks_in_batch:
                task_indices = [i for i, t_id in enumerate(task_ids) if t_id == task_id]
                task_images = images[task_indices]
                
                # Extract and stack labels for the current task
                task_labels_list = [labels[i] for i in task_indices]
                task_labels = torch.stack(task_labels_list, 0)
                
                outputs = model(task_images, task_id=task_id)
                task_name = task_id_to_name[task_id]
                
                if task_name == 'classification':
                    task_metrics[task_id]['Accuracy'].append(calculate_accuracy(task_labels, outputs))
                    task_metrics[task_id]['F1-Score'].append(calculate_f1_score(task_labels, outputs))
                
                elif task_name == 'segmentation':
                    task_metrics[task_id]['Dice'].append(calculate_dice_coefficient(task_labels.to(device), outputs))
                
                elif task_name == 'Regression':
                    task_metrics[task_id]['MAE (pixels)'].append(calculate_mae(task_labels, outputs))
                
                elif task_name == 'detection':
                    # Logic to extract best bounding box from grid prediction
                    batch_size, _, h, w = outputs.shape
                    scores = outputs[:, 4, :, :].view(batch_size, -1) 
                    _, best_indices = torch.max(scores, dim=1)
                    
                    best_h = best_indices // w
                    best_w = best_indices % w
                    
                    final_boxes = torch.zeros((batch_size, 4), device=device)
                    for i in range(batch_size):
                        final_boxes[i] = outputs[i, :4, best_h[i], best_w[i]]
                    
                    task_metrics[task_id]['IoU'].append(calculate_iou(task_labels, final_boxes))

    results = []
    sorted_task_ids = sorted(list(task_id_to_name.keys()))
    for task_id in sorted_task_ids:
        if task_id in task_metrics:
            task_name = task_id_to_name[task_id]
            result_row = {'Task ID': task_id, 'Task Name': task_name}
            for metric_name, values in task_metrics[task_id].items():
                result_row[metric_name] = np.mean(values)
            results.append(result_row)
    return pd.DataFrame(results)