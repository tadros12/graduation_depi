# Preprocessing Pipeline — Sentinel-2 Land Cover Classification

## Overview

The preprocessing pipeline transforms raw Sentinel-2 GeoTIFF satellite imagery into clean, model-ready tensors. It covers five stages: data loading, spectral normalization, spatial standardization, data augmentation, and feature engineering.

---

## Dataset

| Property | Value |
|----------|-------|
| Total files | 719 GeoTIFF files |
| Total size | ~3.1 GB |
| Bands per file | 13 (12 spectral + 1 label) |
| Native spatial dimensions | ~501–502 rows × 545–584 cols |
| Resized to | 256 × 256 |
| Total pixels (approx.) | 205 million+ |

### Geographical Regions

| Region | Images | Notes |
|--------|--------|-------|
| CairoUniv | 125 | Urbanized delta, vegetation pockets |
| IconicTower | 124 | New Administrative Capital, desert/roads |
| SiwaOasis | 123 | Western Desert oasis with lakes/agriculture |
| Alexandria | 70 | Coastal city |
| KarnakLuxor | 70 | Upper Egypt, Nile river |
| Mansoura | 70 | Nile Delta agricultural center |
| PhilaeAswan | 69 | Granite desert, Nile/Lake Nasser |
| HawaraFayoum | 68 | Fayoum Depression agricultural oasis |

### Spectral Bands (Bands 1–12)

| Band Index | Sentinel-2 Band | Wavelength | Description |
|------------|-----------------|------------|-------------|
| 0 | B01 | 443 nm | Coastal Aerosol |
| 1 | B02 | 490 nm | Blue |
| 2 | B03 | 560 nm | Green |
| 3 | B04 | 665 nm | Red |
| 4 | B05 | 705 nm | Vegetation Red Edge |
| 5 | B06 | 740 nm | Vegetation Red Edge |
| 6 | B07 | 783 nm | Vegetation Red Edge |
| 7 | B08 | 842 nm | NIR (Near-Infrared) |
| 8 | B8A | 865 nm | Narrow NIR |
| 9 | B09 | 945 nm | Water Vapour |
| 10 | B11 | 1610 nm | SWIR |
| 11 | B12 | 2190 nm | SWIR |

### Class Distribution (pixel-level)

| Class | Label | Pixel Share | Inverse Weight |
|-------|-------|-------------|----------------|
| 0 | Trees / Forest | 0.61% | 163.93 |
| 1 | Agriculture / Vegetation | 22.22% | 4.50 |
| 2 | Desert / Bare Land | 51.80% | 1.93 |
| 3 | Water Bodies | 8.37% | 11.94 |
| 4 | Urban / Roads | 17.00% | 5.88 |

The dataset is heavily imbalanced — Desert alone accounts for over half of all pixels, while Trees cover less than 1%.

---

## Stage 1 — Data Loading

**File**: `src/dataset.py`

Each GeoTIFF is read with `rasterio`. The file contains 13 bands stacked along the first axis:

- Bands 0–11 → spectral image (float32)
- Band 12 → semantic label mask (int)

```python
with rasterio.open(filepath) as src:
    data = src.read()          # shape: (13, H, W)

image = data[:12].astype(np.float32)   # spectral bands
mask  = data[12].astype(np.int64)      # class labels
```

---

## Stage 2 — Spectral Normalization

**File**: `src/dataset.py` — lines 34–37

Sentinel-2 Digital Numbers (DN) range from 0 to ~10,000 (top-of-atmosphere reflectance scaled by 10,000). They are brought to [0, 1]:

```
image_normalized = clip(image / 10000.0, 0.0, 1.0)
```

The `clip` call removes any out-of-range sensor artifacts before the values enter the model.

---

## Stage 3 — Spatial Standardization (Resizing)

**File**: `src/dataset.py` — lines 43–47

Native image sizes vary slightly across regions (501–502 × 545–584 pixels). All images are resized to a uniform **256 × 256** to enable batch processing.

Two different interpolation modes are used deliberately:

| Data type | Interpolation | Reason |
|-----------|---------------|--------|
| Spectral bands | Bilinear | Preserves continuous reflectance gradients |
| Label mask | Nearest-neighbor | Prevents fractional class values |

Using bilinear on the mask would produce blended values (e.g., 1.7) that do not correspond to any real class.

---

## Stage 4 — Data Augmentation

**File**: `src/dataset.py` — lines 52–67

Augmentations are applied at training time only. Each transform is applied identically to the image tensor and the mask tensor so spatial alignment is preserved.

| Transform | Probability | Details |
|-----------|-------------|---------|
| Horizontal flip | 50% | Left–right mirror |
| Vertical flip | 50% | Top–bottom mirror |
| Random 90° rotation | 25% each | 0°, 90°, 180°, or 270° |

These augmentations are appropriate for satellite imagery because land cover has no inherent orientation — a field photographed upside-down is still a field.

---

## Stage 5 — Feature Engineering

**File**: `src/features.py`

### 5a. NDVI — Normalized Difference Vegetation Index

NDVI uses the Red (B04, band index 3) and NIR (B08, band index 7) channels:

```
NDVI = (NIR - Red) / (NIR + Red)
```

Division-by-zero pixels (where NIR + Red = 0) are handled explicitly:

```python
valid_mask = denominator > 0
ndvi[valid_mask] = (nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]
```

Output range: [−1, 1]  
High values → dense vegetation; negative values → water or cloud.

### 5b. PCA — Dimensionality Reduction

**Goal**: Compress 12 spectral bands into 3 principal components that retain ≥ 95% of the explained variance, for visualization and analysis.

**Steps**:

1. Reshape image from `(12, H, W)` → `(H×W, 12)` (one row per pixel)
2. Z-score standardization per band:
   ```
   X_std = (X - mean) / std
   ```
3. Fit `sklearn.decomposition.PCA(n_components=3)`
4. Transform pixels to 3D PCA space
5. Reshape back to `(3, H, W)`

The three components are mapped to RGB channels to produce a false-color composite that highlights land cover variation.

### 5c. Mutual Information — Band Importance Analysis

**Goal**: Rank all 12 spectral bands by how much information they carry about the class label.

**Steps**:

1. Sample pixels from a subset of images
2. Normalize each band to zero mean and unit variance:
   ```python
   X_norm = (X - X.mean(axis=0)) / X_std   # X_std clamped ≥ 1 to avoid /0
   ```
3. Call `sklearn.feature_selection.mutual_info_classif(X_norm, y)`
4. Rank bands by MI score

This identifies which bands (e.g., NIR, Red Edge) are most discriminative for the five land cover classes and informs future channel selection decisions.

---

## Stage 6 — Class Imbalance Mitigation

**File**: `src/train.py` — lines 125–128

Because Desert pixels dominate (~52%) and Tree pixels are rare (~0.6%), a standard cross-entropy loss would push the model to predict "Desert" for everything and still achieve high accuracy.

The solution is **inverse-frequency weighting**:

```python
proportions = np.array([0.0061, 0.2222, 0.5180, 0.0837, 0.1700])
weights = 1.0 / proportions
weights = weights / weights.sum() * 5.0     # normalize to num_classes
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
```

Rare classes receive a proportionally higher loss penalty, forcing the model to learn their features rather than ignoring them.

---

## Stratified Data Splitting

**File**: `src/dataset.py` — `get_stratified_splits()`

Files are split **by geographical region**, not randomly, to ensure that every region is represented in training, validation, and test sets.

| Split | Proportion | Approx. images |
|-------|------------|----------------|
| Train | 70% | ~504 |
| Validation | 15% | ~108 |
| Test | 15% | ~107 |

`sklearn.model_selection.train_test_split` with `stratify=regions` is used under the hood.

---

## Preprocessing Summary

| Step | Where | Input → Output | Method |
|------|-------|----------------|--------|
| Load GeoTIFF | `dataset.py` | 13-band raster file | `rasterio.open().read()` |
| Spectral normalization | `dataset.py:34–37` | DN [0, 10000] → [0, 1] | Divide by 10000, clip |
| Resize bands | `dataset.py:43–45` | Variable H×W → 256×256 | Bilinear interpolation |
| Resize labels | `dataset.py:46–47` | Variable H×W → 256×256 | Nearest-neighbor |
| Augmentation | `dataset.py:52–67` | 256×256 tensor | Random flip + rotation |
| NDVI | `features.py:7–16` | Red + NIR bands → index | (NIR−Red)/(NIR+Red) |
| PCA | `features.py:18–46` | 12 bands → 3 components | Standardize + sklearn PCA |
| MI band ranking | `features.py:74–82` | 12 bands + labels → scores | Standardize + mutual_info_classif |
| Class weighting | `train.py:125–128` | Pixel proportions → weights | Inverse frequency, normalized |
| Stratified split | `dataset.py` | 719 files → train/val/test | By region, 70/15/15 |
