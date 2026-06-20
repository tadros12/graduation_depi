import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import Sentinel2SegmentationDataset, get_stratified_splits
from src.model import UNet

def compute_iou(preds, targets, num_classes=5):
    """
    Computes Intersection-over-Union (IoU) for each class and their mean.
    preds: Tensor of shape (N, H, W) containing predicted class indices.
    targets: Tensor of shape (N, H, W) containing ground-truth class indices.
    """
    ious = []
    # Avoid double-counting pixels or division by zero
    for c in range(num_classes):
        pred_mask = (preds == c)
        target_mask = (targets == c)
        
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        
        if union == 0:
            # If there are no pixels of this class in ground truth and prediction, IoU is NaN
            # We exclude it from the mean
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
            
    return np.array(ious)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)  # Shape: (batch_size, num_classes, H, W)
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * images.size(0)
        
        # Calculate pixel accuracy
        preds = torch.argmax(outputs, dim=1)
        correct_pixels += (preds == masks).sum().item()
        total_pixels += masks.numel()
        
    return epoch_loss / len(loader.dataset), correct_pixels / total_pixels

def validate_epoch(model, loader, criterion, device, num_classes=5):
    model.eval()
    epoch_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()
            
            # Save for global IoU computation
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())
            
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    class_ious = compute_iou(all_preds, all_targets, num_classes)
    mean_iou = np.nanmean(class_ious)
    pixel_acc = correct_pixels / total_pixels
    val_loss = epoch_loss / len(loader.dataset)
    
    return val_loss, pixel_acc, mean_iou, class_ious

def run_training_pipeline(dataset_dir, checkpoint_dir, epochs=10, batch_size=8, lr=1e-4, device=None):
    """
    Main training pipeline.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 1. Stratified geographical splitting of the dataset
    train_paths, val_paths, test_paths = get_stratified_splits(dataset_dir)
    
    # 2. Instantiate Datasets and DataLoaders
    train_dataset = Sentinel2SegmentationDataset(train_paths, augment=True)
    val_dataset = Sentinel2SegmentationDataset(val_paths, augment=False)
    test_dataset = Sentinel2SegmentationDataset(test_paths, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 3. Instantiate model
    model = UNet(n_channels=12, n_classes=5).to(device)
    
    # 4. Handle class imbalance with weighted loss
    # Class proportions from EDA: Class 0: 0.61%, Class 1: 22.22%, Class 2: 51.80%, Class 3: 8.37%, Class 4: 17.00%
    # We use inverse frequency weights
    proportions = np.array([0.0061, 0.2222, 0.5180, 0.0837, 0.1700])
    weights = 1.0 / proportions
    weights = weights / weights.sum() * 5.0  # Normalize to sum to 5 (num_classes)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"\nApplying class weights to loss function: {weights}")
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_iou = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_miou': []
    }
    
    print("\nStarting Training Loop...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_miou, val_class_ious = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Log progress
        print(f"Epoch {epoch:02d}/{epochs:02d} | Time: {epoch_time:.1f}s")
        print(f"  Train - Loss: {train_loss:.4f}, Pixel Acc: {train_acc*100:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Pixel Acc: {val_acc*100:.2f}%, mIoU: {val_miou*100:.2f}%")
        print(f"  Val class IoUs - C0: {val_class_ious[0]*100:.1f}%, C1: {val_class_ious[1]*100:.1f}%, C2: {val_class_ious[2]*100:.1f}%, C3: {val_class_ious[3]*100:.1f}%, C4: {val_class_ious[4]*100:.1f}%")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_miou'].append(val_miou)
        
        # Save best model checkpoint
        if val_miou > best_val_iou:
            best_val_iou = val_miou
            checkpoint_path = os.path.join(checkpoint_dir, "best_unet_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou,
                'val_class_ious': val_class_ious,
                'class_weights': weights
            }, checkpoint_path)
            print(f"  ---> Saved new best model checkpoint to {checkpoint_path} (mIoU: {best_val_iou*100:.2f}%)")
            
    # 5. Evaluate on test set using the best checkpoint
    print("\nTraining completed! Loading best model for evaluation on Test Set...")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_unet_model.pth")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from epoch {checkpoint['epoch']} with val mIoU {checkpoint['val_miou']*100:.2f}%")
        
    test_loss, test_acc, test_miou, test_class_ious = validate_epoch(model, test_loader, criterion, device)
    print("\n================== TEST SET EVALUATION ==================")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Pixel Accuracy: {test_acc*100:.2f}%")
    print(f"Test Mean IoU (mIoU): {test_miou*100:.2f}%")
    print(f"Test Class IoUs:")
    classes = ["Trees/Forest (Class 0)", "Agriculture/Veg (Class 1)", "Desert (Class 2)", "Water (Class 3)", "Urban/Roads (Class 4)"]
    for cls_name, iou_val in zip(classes, test_class_ious):
        print(f"  {cls_name}: {iou_val*100:.2f}%")
    print("=========================================================")
    
    return history, test_class_ious
