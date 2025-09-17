import torch.nn as nn
import torchvision.models as models


def build_model(pretrained=True, num_classes=2):
    """Build a ResNet18 model. Prefer torchvision `weights=` API when available
    for compatibility with newer torchvision versions. Falls back to `pretrained=`
    for older versions.
    """
    # torchvision 0.13+ introduced `weights=` deprecation of `pretrained`
    try:
        # Try using the new weights enum API
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        # Fallback for older torchvision versions
        model = models.resnet18(pretrained=pretrained)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


if __name__ == "__main__":
    m = build_model()
    print(m)
