import numpy as np
import rasterio
import os
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

def calculate_ndvi(red_band, nir_band):
    """
    Computes the Normalized Difference Vegetation Index (NDVI) for Red and NIR bands.
    NDVI = (NIR - Red) / (NIR + Red)
    """
    denominator = nir_band + red_band
    ndvi = np.zeros_like(nir_band, dtype=np.float32)
    valid_mask = denominator > 0
    ndvi[valid_mask] = (nir_band[valid_mask] - red_band[valid_mask]) / denominator[valid_mask]
    return ndvi

def apply_pca_to_image(image_data, n_components=3):
    """
    Applies Principal Component Analysis (PCA) across the spectral bands of a single image.
    Args:
        image_data (np.ndarray): Spectral image data of shape (bands, height, width).
        n_components (int): Number of principal components to keep.
    Returns:
        pca_image (np.ndarray): Image of shape (n_components, height, width).
        explained_variance (np.ndarray): Ratio of variance explained by each component.
    """
    bands, height, width = image_data.shape
    # Reshape image to (pixels, bands)
    pixels = image_data.reshape(bands, -1).T
    
    # Standardize features
    pixels_mean = pixels.mean(axis=0)
    pixels_std = pixels.std(axis=0)
    # Avoid division by zero
    pixels_std[pixels_std == 0] = 1.0
    pixels_norm = (pixels - pixels_mean) / pixels_std
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pixels_pca = pca.fit_transform(pixels_norm)
    
    # Reshape back to (n_components, height, width)
    pca_image = pixels_pca.T.reshape(n_components, height, width)
    
    return pca_image, pca.explained_variance_ratio_

def analyze_band_importance(dataset_dir, sample_files, n_pixels_per_file=2000):
    """
    Computes mutual information between each spectral band and the land cover labels.
    This quantifies the information shared between individual bands and the classification target.
    """
    X_list = []
    y_list = []
    
    for f in sample_files:
        file_path = os.path.join(dataset_dir, f)
        with rasterio.open(file_path) as src:
            data = src.read()  # (13, H, W)
            bands_data = data[:12].reshape(12, -1).T  # (pixels, 12)
            labels_data = data[12].reshape(-1)        # (pixels,)
            
            # Sample a subset of pixels to keep computation fast
            num_pixels = len(labels_data)
            sample_size = min(n_pixels_per_file, num_pixels)
            indices = np.random.choice(num_pixels, sample_size, replace=False)
            
            X_list.append(bands_data[indices])
            y_list.append(labels_data[indices])
            
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).astype(int)
    
    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    # Calculate Mutual Information for each spectral band
    print("Computing mutual information between bands and target labels (this might take a few seconds)...")
    mi_scores = mutual_info_classif(X_norm, y, random_state=42)
    
    band_names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    band_mi = {name: score for name, score in zip(band_names, mi_scores)}
    
    # Sort bands by mutual information score
    sorted_band_mi = sorted(band_mi.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_band_mi
