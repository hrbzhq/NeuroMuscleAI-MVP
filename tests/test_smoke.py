import os
import sys
from pathlib import Path
import torch
from PIL import Image
# ensure project root is importable when running pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import build_model
from utils import grad_cam, overlay_heatmap
import numpy as np


def test_build_model_forward():
    model = build_model(pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape[0] == 1
    assert out.shape[1] == 2


def test_gradcam_and_overlay(tmp_path):
    model = build_model(pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    # grad_cam should return a heatmap of shape (H, W)
    heatmap = grad_cam(model, x)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)

    # overlay_heatmap should accept a PIL image and return a PIL image
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    overlay = overlay_heatmap(img, heatmap)
    assert overlay.size == img.size
    # Save to tmp to ensure saving works
    outp = tmp_path / "overlay_test.jpg"
    overlay.save(outp)
    assert outp.exists()
