import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_training_history(history, save_path):
    """
    Plots the training and validation loss, accuracy, and mIoU history.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='royalblue', marker='o')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', color='crimson', marker='s')
    axes[0].set_title('Loss History')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    # 2. Pixel Accuracy
    axes[1].plot(epochs, [x*100 for x in history['train_acc']], label='Train Acc', color='royalblue', marker='o')
    axes[1].plot(epochs, [x*100 for x in history['val_acc']], label='Val Acc', color='crimson', marker='s')
    axes[1].set_title('Pixel Accuracy History (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()
    
    # 3. Mean IoU
    axes[2].plot(epochs, [x*100 for x in history['val_miou']], label='Val mIoU', color='forestgreen', marker='^')
    axes[2].set_title('Validation Mean IoU History (%)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('mIoU (%)')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training history plot to {save_path}")


def visualize_predictions(model, dataset, device, save_dir, num_samples=3):
    """
    Runs model inference on random samples from the dataset and saves
    plots showing True Color RGB, Ground Truth Mask, and Predicted Mask side-by-side.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    classes = ["Trees/Forest (0)", "Agriculture/Veg (1)", "Desert (2)", "Water (3)", "Urban/Roads (4)"]
    cmap = plt.get_cmap('tab10')
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image_tensor, mask_tensor = dataset[idx]
            
            # Add batch dimension and move to device
            inputs = image_tensor.unsqueeze(0).to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            
            # Ground truth and input RGB
            image_np = image_tensor.numpy()
            mask_np = mask_tensor.numpy()
            
            # Reconstruct RGB for visualization
            # Bands in image_tensor are scaled to [0,1], index 3=Red, 2=Green, 1=Blue
            red = image_np[3]
            green = image_np[2]
            blue = image_np[1]
            
            # Normalization helper
            def norm(band):
                # Simple stretch to increase visibility
                b_min, b_max = np.percentile(band, [2, 98])
                if b_max - b_min > 0:
                    return np.clip((band - b_min) / (b_max - b_min), 0.0, 1.0)
                return np.zeros_like(band)
                
            rgb = np.stack([norm(red), norm(green), norm(blue)], axis=-1)
            
            # Plotting
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # True Color RGB
            axes[0].imshow(rgb)
            axes[0].set_title("True Color RGB (B4, B3, B2)")
            axes[0].axis('off')
            
            # Ground Truth Mask
            im_gt = axes[1].imshow(mask_np, cmap=cmap, vmin=-0.5, vmax=4.5)
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis('off')
            
            # Predicted Mask
            im_pred = axes[2].imshow(preds, cmap=cmap, vmin=-0.5, vmax=4.5)
            axes[2].set_title("Model Prediction")
            axes[2].axis('off')
            
            # Add discrete colorbar
            cbar = fig.colorbar(im_pred, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.08, ticks=[0, 1, 2, 3, 4])
            cbar.ax.set_xticklabels(classes)
            
            plt.suptitle(f"Sample Inference Prediction (Dataset Index: {idx})", fontsize=14)
            
            out_path = os.path.join(save_dir, f"prediction_sample_{i+1}.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved prediction visualization to {out_path}")


def plot_spectral_signatures(class_signatures, save_path):
    """
    Plots the average spectral signatures (bands B1 to B12) for the 5 classes.
    class_signatures: dict mapping class integer to a list/array of 12 values.
    """
    band_names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    class_names = {
        0: "Trees/Forest",
        1: "Agriculture/Vegetation",
        2: "Desert",
        3: "Water",
        4: "Urban/Roads"
    }
    colors = {
        0: "forestgreen",
        1: "limegreen",
        2: "orange",
        3: "dodgerblue",
        4: "dimgray"
    }
    
    plt.figure(figsize=(12, 6))
    
    for c, sig in class_signatures.items():
        if sig is not None:
            plt.plot(band_names, sig, label=class_names[c], color=colors[c], marker='o', linewidth=2.5, markersize=8)
            
    plt.title("Sentinel-2 Spectral Signatures of Land Types in Egypt", fontsize=14, fontweight='bold')
    plt.xlabel("Spectral Bands", fontsize=12)
    plt.ylabel("Mean Reflectance (Digital Numbers)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved spectral signatures plot to {save_path}")
