import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# 简单的图像预处理函数
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def _find_last_conv_module(model):
    # 找到模型中最后一个 Conv2d 层（简单而稳健的方法）
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def grad_cam(model, input_tensor, class_idx=None):
    """
    对单张 input_tensor (1,C,H,W) 计算 Grad-CAM 热力图并返回与输入同尺寸的 numpy heatmap (H,W) (值 0-1)
    """
    model.eval()

    # 用 hook 获取最后 conv 的特征图和梯度
    activations = None
    gradients = None

    # 尝试定位最后的 conv 层
    target_conv = _find_last_conv_module(model)
    if target_conv is None:
        raise RuntimeError("No Conv2d layer found in model for Grad-CAM")

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        # grad_out 是一个 tuple
        gradients = grad_out[0].detach()

    fh = target_conv.register_forward_hook(forward_hook)
    bh = target_conv.register_full_backward_hook(backward_hook)

    # 前向
    output = model(input_tensor)
    if class_idx is None:
        class_idx = int(output.argmax(dim=1).item())

    # 反向：以预测类别分数为目标
    model.zero_grad()
    score = output[0, class_idx]
    score.backward(retain_graph=True)

    # 解除 hook
    fh.remove()
    bh.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Failed to obtain activations or gradients for Grad-CAM")

    # 计算 channel-wise 权重：对梯度做全局平均池化
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # shape (N, C, 1, 1)

    # 加权求和 activations
    weighted_activations = (weights * activations).sum(dim=1, keepdim=True)  # (N,1,H,W)
    heatmap = torch.relu(weighted_activations[0, 0])

    # 归一化并 resize 到输入尺寸
    heatmap_np = heatmap.cpu().numpy()
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)

    # 调整到 (H,W) 与输入一致：输入 transform 会 resize 到 224x224
    heatmap_resized = cv2.resize(heatmap_np, (input_tensor.shape[-1], input_tensor.shape[-2]))
    return heatmap_resized


def overlay_heatmap(pil_image, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """将 heatmap (H,W, 值0-1) 叠加到 PIL 图像上并返回 PIL 图像"""
    img = np.array(pil_image.convert("RGB"))
    h, w = heatmap.shape
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]))

    overlayed = np.uint8(img * (1 - alpha) + heatmap_color * alpha)
    return Image.fromarray(overlayed)


def save_heatmap(heatmap, save_path):
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
