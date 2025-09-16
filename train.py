import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from model import build_model
from utils import grad_cam, overlay_heatmap

# 加载配置（显式使用 UTF-8 编码以兼容含中文的文件）
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

writer = SummaryWriter(log_dir=config.get("log_dir", "./logs"))

# 数据增强与预处理
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

train_path = os.path.join(config["data_path"], "train")
val_path = os.path.join(config["data_path"], "val")
if not os.path.isdir(train_path) or not os.path.isdir(val_path):
    raise FileNotFoundError(f"数据目录未找到。请确保存在: {train_path} 和 {val_path}")

full_dataset = datasets.ImageFolder(train_path, transform=transform)
# Note: loaders will be created later depending on k-fold or single-fold logic

# 设备选择（支持 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 模型与优化器
pretrained = config.get("pretrained", True)
model = build_model(pretrained=pretrained)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-4))
# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

# 可选 wandb 接入
if config.get("use_wandb", False):
    try:
        import wandb

        wandb.init(project=config.get("experiment_name", "NeuroMuscleAI"), config=config)
        wandb.watch(model)
        use_wandb = True
    except Exception as e:
        print("WandB init failed:", e)
        use_wandb = False
else:
    use_wandb = False


# 训练循环
def run_training_fold(train_indices, val_indices, fold_id=0):
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=config.get("batch_size", 16), shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.get("batch_size", 16))

    best_val = -1.0
    patience = config.get("early_stopping_patience", 5)
    stale = 0
    for epoch in range(config.get("epochs", 10)):
        model.train()
        total_loss = 0.0
        correct = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        acc = correct / len(train_subset)
        writer.add_scalar(f"Fold{fold_id}/Loss/train", total_loss, epoch)
        writer.add_scalar(f"Fold{fold_id}/Accuracy/train", acc, epoch)

        # 验证集评估
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                probs = (
                    torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    if outputs.shape[1] > 1
                    else outputs[:, 0].cpu().numpy()
                )
                preds = outputs.argmax(1).cpu().numpy()
                val_loss += criterion(outputs, labels).item()
                all_preds.extend(preds.tolist())
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_acc = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_auc = None
        try:
            if len(set(all_labels)) == 2:
                val_auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            val_auc = None

        writer.add_scalar(f"Fold{fold_id}/Loss/val", val_loss, epoch)
        writer.add_scalar(f"Fold{fold_id}/Accuracy/val", val_acc, epoch)
        writer.add_scalar(f"Fold{fold_id}/F1/val", val_f1, epoch)
        if val_auc is not None:
            writer.add_scalar(f"Fold{fold_id}/AUC/val", val_auc, epoch)

        print(
            f"Fold {fold_id} Epoch {epoch+1}: Train Acc={acc:.4f}, Val Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc}"
        )

    # 混淆矩阵与分类报告
    cm = confusion_matrix(all_labels, all_preds)
    _report = classification_report(all_labels, all_preds, output_dict=True)

        # 绘制混淆矩阵图像并写入 TensorBoard
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix")
        plt.colorbar(im, ax=ax)
        ticks = np.arange(len(set(all_labels)))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        writer.add_figure(f"Fold{fold_id}/ConfusionMatrix", fig, epoch)
        plt.close(fig)

        # 保存分类报告为文本到日志
        report_text = classification_report(all_labels, all_preds)
        report_path = os.path.join(
            config.get("log_dir", "./logs"), f"fold{fold_id}_report_epoch{epoch+1}.txt"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # 记录少量 Grad-CAM 叠加图到 logs 并上传到 wandb
        try:
            sample_count = min(3, len(all_labels))
            for i in range(sample_count):
                # val_subset returns (tensor,label) so we rebuild a PIL image from tensor
                tensor_img = val_subset[i][0]
                pil_img = Image.fromarray((tensor_img.permute(1, 2, 0).numpy() * 255).astype("uint8"))
                inp = tensor_img.unsqueeze(0).to(device)
                hm = grad_cam(model, inp)
                overlay = overlay_heatmap(pil_img, hm)
                save_path = os.path.join(config.get("log_dir", "./logs"), f"fold{fold_id}_gradcam_e{epoch+1}_s{i}.jpg")
                overlay.save(save_path)
                writer.add_image(f"Fold{fold_id}/GradCAM_sample{i}", np.array(overlay).transpose(2, 0, 1), epoch)
                if use_wandb:
                    wandb.log({f"Fold{fold_id}/GradCAM_sample{i}": wandb.Image(save_path)})
        except Exception as e:
            print("Grad-CAM logging skipped:", e)

        # WandB logging
        if use_wandb:
            logd = {
                "epoch": epoch + 1,
                "train_loss": total_loss,
                "train_acc": acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
            if val_auc is not None:
                logd["val_auc"] = val_auc
            wandb.log(logd)

        # 学习率调度（基于验证准确率）
        scheduler.step(val_acc)

        # 检查是否是最佳模型
        if val_acc > best_val:
            best_val = val_acc
            stale = 0
            if config.get("checkpoint_best", True):
                os.makedirs("models", exist_ok=True)
                best_path = os.path.join("models", f"best_model_fold{fold_id}.pth")
                torch.save(model.state_dict(), best_path)
                print("Saved best model to", best_path)
        else:
            stale += 1

        # 早停
        if stale >= patience:
            print(f"Fold {fold_id} early stopping triggered (no improvement for {patience} epochs).")
            break


k_folds = config.get("k_folds", 1)
if k_folds is None or k_folds <= 1:
    # 单折（使用 train/val 分离的逻辑）
    # 使用 full_dataset 的 train/val 目录已经创建，这里直接创建 DataLoaders
    train_loader = DataLoader(
        datasets.ImageFolder(train_path, transform=transform), batch_size=config.get("batch_size", 16), shuffle=True
    )
    val_loader = DataLoader(
        datasets.ImageFolder(val_path, transform=transform), batch_size=config.get("batch_size", 16)
    )
    # 直接调用 run_training_fold 的逻辑 by creating indices
    indices = list(range(len(full_dataset)))
    # if data in train_path only, fallback to simple split
    # Here we use previously saved val_path as separate validation set
    # We'll just run a single fold using train_path data
    # For consistency, run a simple train loop using existing loaders
    best_val = -1.0
    patience = config.get("early_stopping_patience", 5)
    stale = 0
    for epoch in range(config.get("epochs", 10)):
        model.train()
        total_loss = 0.0
        correct = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)

        # 验证
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                probs = (
                    torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    if outputs.shape[1] > 1
                    else outputs[:, 0].cpu().numpy()
                )
                preds = outputs.argmax(1).cpu().numpy()
                val_loss += criterion(outputs, labels).item()
                all_preds.extend(preds.tolist())
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_acc = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_auc = None
        try:
            if len(set(all_labels)) == 2:
                val_auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            val_auc = None

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        if val_auc is not None:
            writer.add_scalar("AUC/val", val_auc, epoch)

        print(f"Epoch {epoch+1}: Train Acc={acc:.4f}, Val Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc}")

        if use_wandb:
            logd = {
                "epoch": epoch + 1,
                "train_loss": total_loss,
                "train_acc": acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
            if val_auc is not None:
                logd["val_auc"] = val_auc
            wandb.log(logd)

        scheduler.step(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            stale = 0
            if config.get("checkpoint_best", True):
                os.makedirs("models", exist_ok=True)
                best_path = os.path.join("models", "best_model.pth")
                torch.save(model.state_dict(), best_path)
                print("Saved best model to", best_path)
        else:
            stale += 1

        if stale >= patience:
            print(f"Early stopping triggered (no improvement for {patience} epochs).")
            break

else:
    # 使用 k-fold 交叉验证
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    indices = list(range(len(full_dataset)))
    for fold_id, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"Running fold {fold_id+1}/{k_folds}")
        # 重新初始化模型与优化器以避免折间信息泄露
        model = build_model(pretrained=pretrained).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-4))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        run_training_fold(train_idx, val_idx, fold_id + 1)


# 保存模型
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), os.path.join("models", "model_final.pth"))
print("Model checkpoint saved to models/model_final.pth")
writer.close()
