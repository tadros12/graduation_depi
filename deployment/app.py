import io
import json
import os
import sys

import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.express as px
import rasterio
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import UNet

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="SentinelVision — Land Cover AI", version="1.0.0")
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "Trees / Forest",
    1: "Agriculture / Vegetation",
    2: "Desert / Bare Soil",
    3: "Water Bodies",
    4: "Urban / Roads",
}

CLASS_COLORS = {
    0: "#22c55e",
    1: "#84cc16",
    2: "#f59e0b",
    3: "#3b82f6",
    4: "#6b7280",
}

# ── Model loading ─────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=12, n_classes=5).to(device)

_ckpt_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_unet_model.pth")
if os.path.exists(_ckpt_path):
    ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅  Model loaded — epoch {ckpt['epoch']}, val mIoU {ckpt['val_miou']*100:.2f}%")
else:
    model.eval()
    print("⚠️   No checkpoint found — running with random weights")

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(file_bytes: bytes) -> torch.Tensor:
    with rasterio.open(io.BytesIO(file_bytes)) as src:
        img = src.read(list(range(1, 13))).astype(np.float32)
    img = np.clip(img / 10_000.0, 0.0, 1.0)
    t = torch.from_numpy(img).unsqueeze(0)
    t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
    return t.squeeze(0)

# ── Plotly figure builders ────────────────────────────────────────────────────

_DARK_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter, sans-serif"),
    margin=dict(l=8, r=8, t=48, b=8),
)

def _to_dict(fig) -> dict:
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))


def build_rgb(img: torch.Tensor) -> dict:
    arr = img.numpy()
    r, g, b = arr[3], arr[2], arr[1]

    def stretch(band):
        lo, hi = np.percentile(band, [2, 98])
        return np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)

    rgb = (np.stack([stretch(r), stretch(g), stretch(b)], axis=-1) * 255).astype(np.uint8)
    fig = px.imshow(rgb)
    fig.update_layout(
        title=dict(text="True Color RGB Composite (B4/B3/B2)", font=dict(color="#e2e8f0", size=13)),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        **_DARK_BASE,
    )
    return _to_dict(fig)


def build_prediction(mask: np.ndarray) -> dict:
    hover = np.vectorize(CLASS_NAMES.get)(mask)
    colorscale = [[i / 4, CLASS_COLORS[i]] for i in range(5)]
    fig = go.Figure(
        go.Heatmap(
            z=mask.tolist(),
            text=hover.tolist(),
            hovertemplate="<b>%{text}</b><extra></extra>",
            colorscale=colorscale,
            showscale=False,
            zmin=0,
            zmax=4,
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=dict(text="Predicted Land Cover Map", font=dict(color="#e2e8f0", size=13)),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        **_DARK_BASE,
    )
    return _to_dict(fig)


def build_bar(pct: dict) -> dict:
    names = [CLASS_NAMES[i] for i in range(5)]
    vals = [pct.get(n, 0.0) for n in names]
    colors = [CLASS_COLORS[i] for i in range(5)]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=vals,
            marker_color=colors,
            marker_line_color="rgba(255,255,255,0.06)",
            marker_line_width=1,
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=11),
        )
    )
    fig.update_layout(
        title=dict(text="Land Cover Area Distribution", font=dict(color="#e2e8f0", size=13)),
        xaxis=dict(tickfont=dict(color="#94a3b8", size=10), gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(
            title=dict(text="Coverage (%)", font=dict(color="#64748b", size=11)),
            tickfont=dict(color="#94a3b8"),
            gridcolor="rgba(255,255,255,0.04)",
        ),
        **{**_DARK_BASE, "margin": dict(l=40, r=20, t=48, b=64)},
    )
    return _to_dict(fig)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "Only GeoTIFF (.tif / .tiff) files are accepted.")

    raw = await file.read()

    try:
        tensor = preprocess(raw)

        with torch.no_grad():
            logits = model(tensor.unsqueeze(0).to(device))
            mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(int)

        unique, counts = np.unique(mask, return_counts=True)
        total = int(mask.size)
        pct = {CLASS_NAMES[int(u)]: round(float(c) / total * 100, 2) for u, c in zip(unique, counts)}
        for n in CLASS_NAMES.values():
            pct.setdefault(n, 0.0)

        return {
            "filename": file.filename,
            "dominant_class": max(pct, key=pct.get),
            "classes_detected": int(len(unique)),
            "total_pixels": total,
            "percentages": pct,
            "charts": {
                "rgb": build_rgb(tensor),
                "prediction": build_prediction(mask),
                "bar": build_bar(pct),
            },
        }

    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
