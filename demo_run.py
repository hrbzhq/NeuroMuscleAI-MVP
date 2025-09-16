import os
import torch
from PIL import Image
import numpy as np
from model import build_model
from utils import grad_cam, overlay_heatmap

# Demo runner: create a synthetic image, run model (no pretraining to avoid download), save overlay
os.makedirs('logs', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = build_model(pretrained=False)
model = model.to(device)

# Load best model if exists (non-fatal)
best_path = os.path.join('models', 'best_model.pth')
if os.path.exists(best_path):
    try:
        model.load_state_dict(torch.load(best_path, map_location=device))
        print('Loaded best model from', best_path)
    except Exception as e:
        print('Failed to load best model:', e)

# create synthetic RGB image 224x224
img = Image.fromarray((np.random.rand(224,224,3)*255).astype('uint8'))
# convert to tensor like transforms.ToTensor()
import torchvision.transforms as T
transform = T.Compose([T.Resize((224,224)), T.ToTensor()])
input_tensor = transform(img).unsqueeze(0).to(device)

# run grad-cam
try:
    hm = grad_cam(model, input_tensor)
    overlay = overlay_heatmap(img, hm)
    save_path = os.path.join('logs', 'demo_overlay.jpg')
    overlay.save(save_path)
    print('Saved demo overlay to', save_path)
except Exception as e:
    print('Grad-CAM demo failed:', e)
