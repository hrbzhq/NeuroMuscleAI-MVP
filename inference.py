import numpy as np
import torch
import torch.nn.functional as F


def predict_with_probs(model: torch.nn.Module, input_tensor: torch.Tensor, temperature: float = 1.0):
    """Run model forward and return (pred_label:int, probs:np.ndarray).

    Args:
        model: torch model in eval mode or will be set to eval inside.
        input_tensor: single-batch tensor (1, C, H, W).
        temperature: softmax temperature > 0.
    Returns:
        pred_label, probs (1D numpy array)
    """
    model.eval()
    with torch.no_grad():
        out = model(input_tensor)
        probs = F.softmax(out / float(temperature), dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    return pred, probs


def entropy_from_probs(probs: np.ndarray) -> float:
    p = np.array(probs, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())
