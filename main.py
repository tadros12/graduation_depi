import os
import torch
import numpy as np
from src.dataset import get_stratified_splits, Sentinel2SegmentationDataset
from src.features import analyze_band_importance, apply_pca_to_image
from src.train import run_training_pipeline
from src.model import UNet
from src.utils import plot_spectral_signatures, plot_training_history, visualize_predictions
import rasterio
import matplotlib.pyplot as plt

def main():
    print("=========================================================================")
    # Print a beautiful ASCII header representing the pipeline
    print("              Sentinel-2 Land Cover Classification Pipeline             ")
    print("                      Region: Egypt - DEPI Project                      ")
    print("=========================================================================\n")
    
    # Paths setup
    dataset_dir = "/run/media/theodoros/E/projects/dataflow__analyizer/depi_project/egypt_s2_diverse_dataset"
    project_dir = "/run/media/theodoros/E/projects/dataflow__analyizer/depi_project"
    figures_dir = os.path.join(project_dir, "reports/figures")
    checkpoint_dir = os.path.join(project_dir, "models")
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # MILESTONE 1: DATA EXPLORATION, PREPROCESSING, AND FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    print("-------------------------------------------------------------------------")
    print("[Milestone 1] Data Exploration, Stratified Splitting, and Spectral Profiles")
    print("-------------------------------------------------------------------------")
    
    # 1. Stratified Split
    print("Performing stratified split by geographical region...")
    train_paths, val_paths, test_paths = get_stratified_splits(dataset_dir)
    
    # 2. Hardcoded/Computed Mean Spectral Profiles from our inspection script
    # This prevents running a heavy double loop again and uses high-quality stats
    print("\nExtracting mean spectral profiles for each land cover type...")
    class_signatures = {
        0: [989.92, 1154.39, 1557.54, 1872.01, 2262.78, 2638.05, 2822.39, 2905.30, 2943.95, 2952.63, 2920.77, 2445.71],  # Trees/Forest
        1: [604.65, 696.70, 1087.43, 1136.53, 1712.72, 2841.65, 3239.96, 3359.20, 3431.43, 3452.87, 2460.33, 1676.72],    # Agriculture
        2: [1529.18, 1912.10, 2669.42, 3476.46, 3836.05, 3929.06, 4041.53, 4114.59, 4103.97, 4146.97, 4820.83, 4363.43],  # Desert
        3: [568.54, 553.66, 828.49, 649.37, 801.28, 788.07, 851.88, 809.23, 853.06, 1219.69, 842.16, 675.76],             # Water
        4: [1172.15, 1418.38, 1870.29, 2280.90, 2628.20, 2968.75, 3138.94, 3224.17, 3260.68, 3303.83, 3419.48, 3018.46]   # Urban/Roads
    }
    
    spectral_plot_path = os.path.join(figures_dir, "spectral_signatures.png")
    plot_spectral_signatures(class_signatures, spectral_plot_path)
    
    # -------------------------------------------------------------------------
    # MILESTONE 2: ADVANCED DATA ANALYSIS & MODEL SELECTION
    # -------------------------------------------------------------------------
    print("\n-------------------------------------------------------------------------")
    print("[Milestone 2 - Part A] Advanced Data Analysis (Mutual Info & PCA)")
    print("-------------------------------------------------------------------------")
    
    # 1. Band Importance Analysis via Mutual Information
    # Using 15 sampled files to compute mutual information between bands and labels
    sample_files = sorted(os.listdir(dataset_dir))
    sample_files = [f for f in sample_files if f.endswith('.tif')]
    # Take a sample across different classes
    sampled_mi_files = sample_files[::45]
    
    mi_scores = analyze_band_importance(dataset_dir, sampled_mi_files)
    print("\nSpectral Band Importance (Sorted by Mutual Information Score):")
    for rank, (band, score) in enumerate(mi_scores, 1):
        print(f"  Rank {rank:02d}: Band {band:<3} | Mutual Info Score: {score:.4f}")
        
    # Plot Band Importance
    plt.figure(figsize=(10, 5))
    bands, scores = zip(*mi_scores)
    plt.bar(bands, scores, color='darkslateblue', edgecolor='black', alpha=0.8)
    plt.title("Sentinel-2 Band Importance for Land Cover Classification", fontsize=12, fontweight='bold')
    plt.xlabel("Spectral Band", fontsize=11)
    plt.ylabel("Mutual Information Score", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    mi_plot_path = os.path.join(figures_dir, "band_importance.png")
    plt.savefig(mi_plot_path, dpi=150)
    plt.close()
    print(f"Saved band importance plot to {mi_plot_path}")
    
    # 2. PCA Dimensionality Reduction
    # Apply PCA to a sample file (e.g. HawaraFayoum) and save components
    pca_sample = "HawaraFayoum_000_30.932_29.252.tif"
    print(f"\nApplying PCA dimensionality reduction on image: {pca_sample}")
    with rasterio.open(os.path.join(dataset_dir, pca_sample)) as src:
        image_data = src.read(list(range(1, 13)))  # Load first 12 spectral bands
        
    pca_image, variance_ratios = apply_pca_to_image(image_data, n_components=3)
    print("Explained Variance Ratio per Principal Component:")
    for comp, var in enumerate(variance_ratios, 1):
        print(f"  PC {comp}: {var*100:.2f}% explained variance")
    print(f"  Total Explained Variance (3 Components): {sum(variance_ratios)*100:.2f}%")
    
    # Normalize components to [0, 1] for visualization
    def norm_comp(comp):
        c_min, c_max = comp.min(), comp.max()
        if c_max - c_min > 0:
            return (comp - c_min) / (c_max - c_min)
        return np.zeros_like(comp)
        
    pca_rgb = np.stack([norm_comp(pca_image[0]), norm_comp(pca_image[1]), norm_comp(pca_image[2])], axis=-1)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(pca_rgb)
    plt.title(f"PCA Color Composite (PC1, PC2, PC3)\nTotal Explained Variance: {sum(variance_ratios)*100:.2f}%", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    pca_plot_path = os.path.join(figures_dir, "pca_composite.png")
    plt.savefig(pca_plot_path, dpi=150)
    plt.close()
    print(f"Saved PCA color composite image to {pca_plot_path}")
    
    # 3. Model Training & Selection
    print("\n-------------------------------------------------------------------------")
    print("[Milestone 2 - Part B] Model Selection & Training Pipeline (U-Net)")
    print("-------------------------------------------------------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run training for 10 epochs.
    # Since we want to train efficiently and demonstrate functionality, 10 epochs is standard.
    history, test_class_ious = run_training_pipeline(
        dataset_dir=dataset_dir,
        checkpoint_dir=checkpoint_dir,
        epochs=10,
        batch_size=8,
        lr=1e-4,
        device=device
    )
    
    # Plot learning curves
    history_plot_path = os.path.join(figures_dir, "training_learning_curves.png")
    plot_training_history(history, history_plot_path)
    
    # Visualize visual prediction maps on test set
    test_dataset = Sentinel2SegmentationDataset(test_paths, augment=False)
    # Reload best model weights for visualization inference
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_unet_model.pth")
    model = UNet(n_channels=12, n_classes=5).to(device)
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    visualize_predictions(model, test_dataset, device, figures_dir, num_samples=3)
    
    print("\n=========================================================================")
    print("                Pipeline Completed Successfully!                ")
    print(f" All outputs and figures saved to: {figures_dir} ")
    print("=========================================================================")

if __name__ == '__main__':
    main()
