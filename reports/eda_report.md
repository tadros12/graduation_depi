# Exploratory Data Analysis (EDA) Report
## Land Cover Classification using Sentinel-2 Satellite Images (Egypt)

---

### 1. Executive Summary
This report summarizes the exploratory data analysis, preprocessing steps, and feature engineering strategies designed for classifying land cover types across Egypt using multi-spectral satellite imagery from the European Space Agency's **Sentinel-2** mission. 

The dataset consists of **719 high-resolution satellite imagery tiles** representing key geographical regions in Egypt. Each image tile contains **12 spectral bands** (from visible bands to shortwave infrared) and a **13th label band** mapping each pixel to one of 5 distinct land cover classes. 

Our analysis reveals a significant class imbalance (with Desert covering over 51.8% of the surface area and Trees/Forest representing just 0.61%), which we address using inverse-frequency loss weighting. Spectral profiling was used to successfully identify the semantic meaning of each numeric class.

---

### 2. Dataset Overview & Composition

#### 2.1 File Counts and Regional Representation
The dataset contains 719 `.tif` image files. The files are prefixed with their corresponding geographical region in Egypt:

| Geographical Region (Prefix) | Number of Images | Description |
| :--- | :--- | :--- |
| **CairoUniv** | 125 | Highly urbanized delta region, containing urban infrastructure and pockets of vegetation. |
| **IconicTower** | 124 | Representing the New Administrative Capital; dominated by desert terrain, asphalt roads, and new construction. |
| **SiwaOasis** | 123 | Oasis in the Western Desert; contains hypersaline lakes (water), agriculture (palms/olives), and sand dunes. |
| **Alexandria** | 70 | Coastal city on the Mediterranean; containing seawater, urban centers, and coastal lagoons. |
| **KarnakLuxor** | 70 | Upper Egypt historic center; contains the Nile river, narrow agricultural bands, and flanking desert. |
| **Mansoura** | 70 | Center of the Nile Delta; highly dominated by intensive agricultural fields and rural towns. |
| **PhilaeAswan** | 69 | Upper Egypt granite desert; contains the Nile river, Lake Nasser, rocky desert, and historical monuments. |
| **HawaraFayoum** | 68 | Fayoum Depression; agricultural oasis connected to Lake Qarun and surrounding desert. |
| **Total** | **719** | **Comprehensive spatial representation of Egypt's major ecosystems.** |

#### 2.2 Spatial Dimensions
The spatial resolution of the tiles varies slightly due to crop borders:
* Heights: **501 to 502 pixels**
* Widths: **545 to 584 pixels**
* Resizing Strategy: All images are resized to a uniform **256 × 256 pixels** during training. Bilinear interpolation is used for spectral bands, and Nearest-Neighbor interpolation is used for the label maps to prevent interpolation artifacts (which would create fractional/invalid class labels).

---

### 3. Class Distribution & Semantic Mapping

Analyzing the 13th band (label band) across all 719 files (comprising over 205 million pixels) revealed **5 integer classes (0 to 4)**. We established their semantic meaning by inspecting their **Mean NDVI** and **Spectral Reflectance Signatures**:

| Numeric Label | Pixel Frequency (%) | Mean NDVI | Distinctive Spectral Behavior | Identified Land Cover Type |
| :---: | :---: | :---: | :--- | :--- |
| **Class 0** | 0.61% | 0.2667 | Moderate NDVI; low visible and moderate NIR/SWIR reflectance. | **Trees / Forest** |
| **Class 1** | 22.22% | 0.5092 | Very high NDVI; high NIR reflectance (B08) and strong red chlorophyll absorption (B04). | **Agriculture / Dense Vegetation** |
| **Class 2** | 51.80% | 0.0879 | Low positive NDVI; extremely high reflectance across all channels, especially SWIR (B11/B12). | **Desert / Bare Land** |
| **Class 3** | 8.37% | 0.0155 | Near-zero NDVI; extremely low reflectance across all channels, especially NIR/SWIR (complete water absorption). | **Water Bodies** |
| **Class 4** | 17.00% | 0.1865 | Low positive NDVI; moderate reflectance in visible and SWIR channels (concrete, asphalt, rooftops). | **Urban Areas / Roads** |

#### Key Insight: Severe Class Imbalance
Desert (Class 2) covers **51.8%** of the pixels in the dataset, representing the dominant landscape of Egypt. Conversely, Trees/Forest (Class 0) covers only **0.61%**. 
* **Impact**: Standard Cross-Entropy loss will cause the model to ignore minority classes.
* **Mitigation**: We compute inverse-frequency weights:
  $$\text{Weight}_c = \frac{1}{\text{Proportion}_c}$$
  These weights are normalized and passed to the PyTorch loss function to penalize errors on minority classes (like Trees and Water) much more heavily.

---

### 4. Spectral Signatures
Sentinel-2 measures solar reflectance across 12 distinct bands. The figure `spectral_signatures.png` plots these values for each class:
1. **Water (Class 3)**: Absorbs almost all electromagnetic radiation; remains flat and low.
2. **Desert (Class 2)**: Reflects energy intensely, peaking in the Shortwave Infrared (SWIR) bands (B11 and B12).
3. **Agriculture (Class 1)**: Shows the classic "red edge" profile: low reflectance in B04 (Red absorption by chlorophyll) and a sharp rise to high reflectance in B08 (NIR scattering by leaf cells).

---

### 5. Preprocessing & Feature Engineering

1. **Spectral Normalization**:
   Raw Sentinel-2 Digital Numbers (DN) represent top-of-atmosphere reflectance scaled by 10,000. We normalize inputs to the range $[0.0, 1.0]$ by dividing by $10,000.0$ and clipping values outside this boundary:
   $$\text{Normalized Band} = \text{clip}\left(\frac{\text{Band}}{10000}, 0.0, 1.0\right)$$

2. **Normalized Difference Vegetation Index (NDVI)**:
   Calculated dynamically using the Red (B4) and Near-Infrared (B8) bands:
   $$\text{NDVI} = \frac{\text{B08} - \text{B04}}{\text{B08} + \text{B04}}$$
   This is highly informative for distinguishing between urban concrete (moderate NDVI) and dense agriculture (high NDVI).

3. **Dimensionality Reduction (PCA)**:
   By standardizing the 12 spectral bands and applying PCA, we can condense the data into 3 principal components that explain over **95%+ of the variance**. Projecting these components as an RGB composite highlights structural differences (soil moisture, urban boundaries, water edges) while reducing the feature space.

4. **Image Augmentation**:
   To prevent overfitting and expand dataset diversity, we apply:
   * Random Horizontal Flips (50% probability)
   * Random Vertical Flips (50% probability)
   * Random 90-degree Rotations (0, 90, 180, 270 degrees)
   Augmentations are applied simultaneously to the image tensor and label map to maintain pixel-perfect alignment.

---

### 6. Modeling Strategy
For pixel-level semantic segmentation, we select a **U-Net** architecture. U-Net's encoder-decoder structure with skip connections is optimal for satellite images because:
1. The **contracting path (encoder)** extracts high-level semantic context (e.g., distinguishing urban vs. agricultural zones).
2. The **expanding path (decoder)** recovers fine-grained spatial boundaries (e.g., narrow roads, canal edges).
3. The **skip connections** transfer precise spatial coordinates directly from early encoder blocks to the decoder, preventing boundary blur.

The network takes a $(12 \times 256 \times 256)$ tensor as input and outputs a $(5 \times 256 \times 256)$ probability logit map, optimized via Weighted Cross-Entropy.
