import torch.nn as nn
import torchvision.models as models


def build_model(pretrained=True, num_classes=2):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


if __name__ == "__main__":
    m = build_model()
    print(m)
