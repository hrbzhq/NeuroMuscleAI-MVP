"""Run a quick batch smoke test using artifacts/test_images.zip and artifacts/test_labels.csv
This script loads the project model, expands the zip, runs inference via inference.predict_with_probs,
and prints accuracy and a few sample predictions. It avoids Streamlit and is reproducible.
"""
import os
import sys
import io
import zipfile
# Ensure project root is on sys.path so imports work when running from tools/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from batch_utils import expand_zip_bytes, parse_labels_csv
from inference import predict_with_probs
from model import build_model
import torch
from PIL import Image
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ARTIFACTS = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
ZIP_PATH = os.path.join(ARTIFACTS, 'test_images.zip')
CSV_PATH = os.path.join(ARTIFACTS, 'test_labels.csv')


def load_image_bytes(b):
    return Image.open(io.BytesIO(b)).convert('RGB')


def pil_to_tensor(img: Image.Image):
    arr = np.array(img).astype(np.float32) / 255.0
    # transpose to C,H,W
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def main():
    assert os.path.exists(ZIP_PATH), f"Missing {ZIP_PATH}"
    assert os.path.exists(CSV_PATH), f"Missing {CSV_PATH}"

    with open(ZIP_PATH, 'rb') as f:
        zbytes = f.read()

    files = expand_zip_bytes(zbytes)
    labels = {}
    with open(CSV_PATH, 'rb') as f:
        labels = parse_labels_csv(f.read())

    model = build_model(pretrained=False)
    model.eval()

    total = 0
    correct = 0
    samples = []
    for name, b in files:
        try:
            img = load_image_bytes(b)
            inp = pil_to_tensor(img)
            pred_label, probs = predict_with_probs(model, inp, temperature=1.0)
            gt = labels.get(name, None)
            samples.append((name, int(pred_label), float(probs[pred_label]), gt))
            if gt is not None:
                total += 1
                if int(pred_label) == int(gt):
                    correct += 1
        except Exception as e:
            print('Error processing', name, e)

    print('Ran', len(files), 'files. Labeled:', total)
    if total > 0:
        print('Accuracy:', correct / total)
    print('\nSample predictions:')
    for s in samples[:10]:
        print(s)


if __name__ == '__main__':
    main()
