import os
import glob
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF

class Sentinel2SegmentationDataset(Dataset):
    """
    Sentinel-2 satellite image dataset for land type semantic segmentation in Egypt.
    Bands 1-12 are spectral bands. Band 13 contains the land cover class labels.
    """
    def __init__(self, file_paths, target_size=(256, 256), augment=False):
        self.file_paths = file_paths
        self.target_size = target_size
        self.augment = augment
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        with rasterio.open(file_path) as src:
            # Read spectral bands (1 to 12)
            # Sentinel-2 raw digital numbers are typically 0 to 10000
            image = src.read(list(range(1, 13))).astype(np.float32)
            
            # Read label band (13)
            mask = src.read(13).astype(np.int64)
            
        # Scale spectral bands to [0, 1] (typical Sentinel-2 DN values are <= 10000)
        # We divide by 10000.0 and clip to [0, 1]
        image = image / 10000.0
        image = np.clip(image, 0.0, 1.0)
        
        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image)  # Shape: (12, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)   # Shape: (1, H, W)
        
        # Resize image and mask to consistent target size
        # Images: bilinear interpolation
        # Masks: nearest neighbor interpolation to preserve discrete class labels
        image_tensor = TF.resize(image_tensor, self.target_size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        mask_tensor = TF.resize(mask_tensor, self.target_size, interpolation=TF.InterpolationMode.NEAREST)
        
        mask_tensor = mask_tensor.squeeze(0)  # Shape: (H, W)
        
        # Apply augmentations (random flips and rotations)
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                image_tensor = TF.hflip(image_tensor)
                mask_tensor = TF.hflip(mask_tensor)
                
            # Random vertical flip
            if torch.rand(1) > 0.5:
                image_tensor = TF.vflip(image_tensor)
                mask_tensor = TF.vflip(mask_tensor)
                
            # Random 90 degree rotations
            rot_choice = torch.randint(0, 4, (1,)).item()
            if rot_choice > 0:
                image_tensor = torch.rot90(image_tensor, rot_choice, [1, 2])
                mask_tensor = torch.rot90(mask_tensor, rot_choice, [0, 1])
                
        return image_tensor, mask_tensor


def get_stratified_splits(dataset_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Splits files into train, validation, and test sets, stratified by geographical region.
    Regions are determined by the filename prefix (e.g., 'Alexandria', 'CairoUniv').
    """
    file_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.tif")))
    
    # Get regions/cities from file names
    regions = [os.path.basename(fp).split('_')[0] for fp in file_paths]
    
    # First split into train and temp (val + test)
    train_paths, temp_paths, train_regions, temp_regions = train_test_split(
        file_paths, regions,
        test_size=(val_ratio + test_ratio),
        stratify=regions,
        random_state=random_seed
    )
    
    # Calculate ratio for val/test relative to temp split
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    
    # Split temp into val and test
    val_paths, test_paths, _, _ = train_test_split(
        temp_paths, temp_regions,
        test_size=relative_test_ratio,
        stratify=temp_regions,
        random_state=random_seed
    )
    
    print(f"Dataset split summary:")
    print(f"  Total files: {len(file_paths)}")
    print(f"  Training: {len(train_paths)} files")
    print(f"  Validation: {len(val_paths)} files")
    print(f"  Testing: {len(test_paths)} files")
    
    # Print regional distribution in splits to verify stratification
    unique_regions = sorted(list(set(regions)))
    print("\nRegional Distribution in Splits:")
    print(f"{'Region':<15} | {'Train':<5} | {'Val':<5} | {'Test':<5} | {'Total':<5}")
    print("-" * 45)
    for reg in unique_regions:
        total_r = regions.count(reg)
        train_r = [fp.split('/')[-1].startswith(reg) for fp in train_paths].count(True)
        val_r = [fp.split('/')[-1].startswith(reg) for fp in val_paths].count(True)
        test_r = [fp.split('/')[-1].startswith(reg) for fp in test_paths].count(True)
        print(f"{reg:<15} | {train_r:<5} | {val_r:<5} | {test_r:<5} | {total_r:<5}")
        
    return train_paths, val_paths, test_paths
