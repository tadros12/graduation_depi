# 🚀 Task 1: Sentinel-2 Land Cover Classifier API & Web UI Handout
**DEPI Project | Milestone 4: Deployment & Web Interface**

This document serves as the implementation handout for building and deploying the **Sentinel-2 Land Cover Semantic Segmentation Web Application**.

---

## 🎯 Task Objective
Deploy the pre-trained deep learning model (`best_unet_model.pth`) as a web application. Users should be able to upload a multi-spectral Sentinel-2 satellite image (`.tif` file) and view:
1. **True Color RGB Composite** (Bands 4, 3, and 2)
2. **Predicted Land Cover Segmentation Map** (color-coded by class: Forest, Agriculture, Desert, Water, and Urban)
3. **Class Area Distribution (%)**

Both visualizations must be interactive, hoverable, and zoomable using **Plotly**.

---

## 📂 Required Folder Structure
Instruct the developers to set up the following directory structure inside the `depi_project` folder:

```text
depi_project/
├── models/
│   └── best_unet_model.pth         # Pre-trained PyTorch weights
├── src/
│   └── model.py                    # UNet architecture class
│   └── dataset.py                  # Data utility logic
├── deployment/
│   ├── app.py                      # Flask App (REST API + Page server)
│   └── templates/
│       └── index.html              # Frontend page template
```

---

## 🛠️ Data Preprocessing & Model Inference Specifications

For every uploaded `.tif` image, the backend performs the following preprocessing operations in memory before executing the model:

1. **Read Spectral Bands**: Read the first 12 bands from the `.tif` file using `rasterio` (shape: `12 × H × W`).
2. **Normalize Reflectance**: Sentinel-2 values typically range between `0` and `10,000`. Scale values to `[0.0, 1.0]` by dividing by `10,000.0` and clipping the result.
3. **Resizing**: Resize the dimensions to a fixed shape of **$256 \times 256$ pixels** using PyTorch's bilinear interpolation.
4. **Batch Dimension**: Add a batch channel to feed the model (shape: `1 × 12 × 256 × 256`).
5. **Inference**: Pass the tensor to the U-Net model to get logits of shape `1 × 5 × 256 × 256`. 
6. **Mask Generation**: Extract the index of the highest logit along the class channel to get the final predictions mask (shape: `256 × 256`).

---

## 💻 Codebase

### 1. The Flask Backend (`deployment/app.py`)
This script handles routing, PyTorch image preprocessing, model predictions, and converts Plotly plots to JSON format for rendering.

```python
import os
import io
import json
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from flask import Flask, request, jsonify, render_template
import plotly
import plotly.graph_objects as go
import plotly.express as px

# Adjust Python path to import source scripts
import sys
sys.path.append(os.path.abspath('..'))
from src.model import UNet

app = Flask(__name__)

# Constants & Class mappings
CLASS_NAMES = {
    0: "Trees / Forest",
    1: "Agriculture / Vegetation",
    2: "Desert / Bare Soil",
    3: "Water Bodies",
    4: "Urban / Roads"
}
CLASS_COLORS = {
    0: "#228B22",  # Forest Green
    1: "#7FFF00",  # Chartreuse / Lime
    2: "#FFA500",  # Orange
    3: "#1E90FF",  # Dodger Blue
    4: "#808080"   # Gray
}

# Load Trained U-Net Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=12, n_classes=5).to(device)
checkpoint_path = "../models/best_unet_model.pth"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded successfully from epoch {checkpoint['epoch']}!")
else:
    print(f"WARNING: Checkpoint not found at {checkpoint_path}. Running with empty weights.")

def preprocess_image(file_bytes):
    """
    Reads 12 bands using rasterio, normalizes, and resizes to 256x256 tensor.
    """
    with rasterio.open(io.BytesIO(file_bytes)) as src:
        image = src.read(list(range(1, 13))).astype(np.float32)
        
    # Scale reflectance from [0, 10000] to [0.0, 1.0]
    image = image / 10000.0
    image = np.clip(image, 0.0, 1.0)
    
    img_tensor = torch.from_numpy(image)
    img_tensor = TF.resize(img_tensor, (256, 256), interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    return img_tensor

def make_plotly_rgb(img_tensor):
    """
    Creates an interactive RGB Plotly image using bands 4 (Red), 3 (Green), and 2 (Blue).
    """
    img_np = img_tensor.numpy()
    red, green, blue = img_np[3], img_np[2], img_np[1]
    
    # 2% percentile stretch
    def stretch(band):
        vmin, vmax = np.percentile(band, [2, 98])
        return np.clip((band - vmin) / (vmax - vmin), 0.0, 1.0) if vmax - vmin > 0 else band
        
    rgb = np.stack([stretch(red), stretch(green), stretch(blue)], axis=-1)
    
    fig = px.imshow(rgb, binary_string=True)
    fig.update_layout(title="True Color RGB Composite", margin=dict(l=10, r=10, t=30, b=10))
    return fig

def make_plotly_prediction(pred_mask):
    """
    Creates an interactive Plotly heatmap representation of the classification map
    with custom discrete colors and hover values.
    """
    hover_text = np.vectorize(CLASS_NAMES.get)(pred_mask)
    colorscale = [[i/4.0, CLASS_COLORS[i]] for i in range(5)]
    
    fig = go.Figure(data=go.Heatmap(
        z=pred_mask,
        text=hover_text,
        hovertemplate="X: %{x}<br>Y: %{y}<br><b>Class: %{text}</b><extra></extra>",
        colorscale=colorscale,
        showscale=False
    ))
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Predicted Classification Map", margin=dict(l=10, r=10, t=30, b=10))
    return fig

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    file_bytes = file.read()
    
    try:
        # Preprocess input
        img_tensor = preprocess_image(file_bytes)
        
        # Run Inference
        inputs = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            
        # Calculate statistics
        unique, counts = np.unique(preds, return_counts=True)
        total_pixels = preds.size
        
        percentages = {CLASS_NAMES[int(u)]: float(c) / total_pixels * 100 for u, c in zip(unique, counts)}
        for name in CLASS_NAMES.values():
            if name not in percentages:
                percentages[name] = 0.0
                
        # Generate Plotly figures
        fig_rgb = make_plotly_rgb(img_tensor)
        fig_pred = make_plotly_prediction(preds)
        
        fig_bar = px.bar(
            x=list(percentages.keys()),
            y=list(percentages.values()),
            labels={'x': 'Land Cover Type', 'y': 'Percentage (%)'},
            color=list(percentages.keys()),
            color_discrete_map=CLASS_NAMES
        )
        for i, bar in enumerate(fig_bar.data):
            class_idx = list(CLASS_NAMES.values()).index(bar.name)
            bar.marker.color = CLASS_COLORS[class_idx]
            
        fig_bar.update_layout(title="Class Area Distribution (%)", margin=dict(l=10, r=10, t=30, b=10))
        
        # Convert figures to JSON JSON-serializable structures
        rgb_json = json.dumps(fig_rgb, cls=plotly.utils.PlotlyJSONEncoder)
        pred_json = json.dumps(fig_pred, cls=plotly.utils.PlotlyJSONEncoder)
        bar_json = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            "percentages": percentages,
            "charts": {
                "rgb": json.loads(rgb_json),
                "prediction": json.loads(pred_json),
                "bar": json.loads(bar_json)
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

---

### 2. The HTML Frontend (`deployment/templates/index.html`)
This file handles the client-side user interface. It sends the uploaded image via AJAX and renders the interactive Plotly visualizations.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentinel-2 Land Cover Classifier</title>
    <!-- Tailwind CSS for sleek UI styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Include Plotly JS library -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
    
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <header class="text-center mb-10">
            <h1 class="text-4xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-500">
                Sentinel-2 Land Cover Classifier
            </h1>
            <p class="text-gray-400 mt-2">Upload a multi-spectral image file to run semantic segmentation using U-Net</p>
        </header>

        <!-- Upload Box -->
        <div class="max-w-xl mx-auto bg-gray-800 p-6 rounded-lg border border-gray-700 shadow-lg mb-10">
            <form id="uploadForm" class="flex flex-col items-center">
                <label class="w-full flex flex-col items-center px-4 py-6 bg-gray-900 text-gray-300 rounded-lg border-2 border-dashed border-gray-600 cursor-pointer hover:border-emerald-400 hover:text-emerald-400 transition">
                    <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
                    <span class="mt-2 text-sm">Select a .tif image file</span>
                    <input type='file' id="fileInput" accept=".tif" class="hidden" />
                </label>
                <div id="fileName" class="text-xs text-gray-400 mt-2"></div>
                <button type="submit" class="mt-4 w-full bg-emerald-500 hover:bg-emerald-600 text-white font-bold py-2 px-4 rounded transition">
                    Run Classification
                </button>
            </form>
        </div>

        <!-- Loader -->
        <div id="loading" class="hidden text-center text-emerald-400 mb-8 animate-pulse font-semibold">
            Uploading image and executing U-Net model on GPU...
        </div>

        <!-- Dashboard Outputs -->
        <div id="results" class="hidden grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- True Color RGB Chart -->
            <div class="bg-gray-800 p-4 rounded-lg border border-gray-700">
                <div id="rgbPlot" class="w-full h-[400px]"></div>
            </div>

            <!-- Prediction Heatmap Chart -->
            <div class="bg-gray-800 p-4 rounded-lg border border-gray-700">
                <div id="predPlot" class="w-full h-[400px]"></div>
            </div>

            <!-- Class Percentages Bar Chart -->
            <div class="bg-gray-800 p-4 rounded-lg border border-gray-700 lg:col-span-2">
                <div id="barPlot" class="w-full h-[300px]"></div>
            </div>
        </div>
    </div>

    <!-- AJAX script to talk to Flask REST API -->
    <script>
        const fileInput = document.getElementById('fileInput');
        const fileNameDiv = document.getElementById('fileName');
        const form = document.getElementById('uploadForm');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');

        // Show filename when chosen
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                fileNameDiv.textContent = `Selected: ${fileInput.files[0].name}`;
            }
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (fileInput.files.length === 0) {
                alert("Please select a .tif file first.");
                return;
            }

            loading.classList.remove('hidden');
            results.classList.add('hidden');

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    loading.classList.add('hidden');
                    results.classList.remove('hidden');
                    
                    // Render Plotly charts dynamically
                    Plotly.newPlot('rgbPlot', data.charts.rgb.data, data.charts.rgb.layout, {responsive: true});
                    Plotly.newPlot('predPlot', data.charts.prediction.data, data.charts.prediction.layout, {responsive: true});
                    Plotly.newPlot('barPlot', data.charts.bar.data, data.charts.bar.layout, {responsive: true});
                } else {
                    alert(`Error: ${data.error}`);
                    loading.classList.add('hidden');
                }
            } catch (err) {
                alert(`Network error: ${err.message}`);
                loading.classList.add('hidden');
            }
        });
    </script>
</body>
</html>
```

---

## 🏁 Step-by-Step Instructions to Run and Test

1. **Install Dependencies**:
   Open a terminal and install the required packages:
   ```bash
   pip install Flask plotly flask-cors rasterio numpy torch torchvision
   ```
2. **Navigate and Run**:
   Go to the `deployment` directory and start the Flask server:
   ```bash
   cd depi_project/deployment
   python app.py
   ```
3. **Open in Browser**:
   Open your web browser and navigate to `http://localhost:5000`.
4. **Test with Sample Data**:
   Upload a sample `.tif` image from the extracted dataset directory (`depi_project/egypt_s2_diverse_dataset/`) and click "Run Classification" to see the U-Net model automatically generate interactive classification maps and bar charts.
