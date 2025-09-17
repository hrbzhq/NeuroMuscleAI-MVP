import torch
import numpy as np
from torch import nn

from inference import predict_with_probs, entropy_from_probs


class DummyModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 4 * 4, num_classes)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(4, 4), mode='bilinear', align_corners=False)
        x = self.flatten(x)
        return self.fc(x)


def test_predict_with_probs_shape_and_sum():
    model = DummyModel(num_classes=4)
    # create a single 3x8x8 input batch
    x = torch.randn(1, 3, 8, 8)
    pred, probs = predict_with_probs(model, x, temperature=1.0)
    assert isinstance(pred, int)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (4,)
    # probabilities should sum to ~1
    assert abs(probs.sum() - 1.0) < 1e-6


def test_entropy_monotonicity():
    # entropy should be 0 for one-hot and larger for uniform
    one_hot = np.array([1.0, 0.0, 0.0])
    uniform = np.array([1/3, 1/3, 1/3])
    e_one = entropy_from_probs(one_hot)
    e_uni = entropy_from_probs(uniform)
    assert e_one >= 0.0
    assert e_uni > e_one
