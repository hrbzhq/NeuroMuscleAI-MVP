<!-- 简体中文 README -->
<h1 align="center">NeuroMuscleAI-MVP</h1>

<p align="center">
  <a href="README.md">English</a> &nbsp;|&nbsp; <strong>简体中文</strong> &nbsp;|&nbsp; <a href="#">繁體中文</a> &nbsp;|&nbsp; <a href="#">日本語</a> &nbsp;|&nbsp; <a href="#">한국어</a>
</p>

---

## 概要

NeuroMuscleAI-MVP 是一个用于肌电/神经影像小样本训练与可视化的演示级项目：包含训练脚本、Grad-CAM 可视化、Streamlit 演示与多语言支持。

## 快速开始

运行演示：

```powershell
python demo_run.py --input examples/sample.jpg --output out.png
streamlit run app.py
```

### 安装（示例）

在 Windows PowerShell 中：

```powershell
git clone https://github.com/hrbzhq/NeuroMuscleAI-MVP.git
cd NeuroMuscleAI-MVP
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果你使用 Linux / macOS，请使用对应的虚拟环境激活命令。

### 配置

- 若存在 `config.yaml`，在运行训练或演示前调整学习率、批次大小等。示例字段：`learning_rate`, `batch_size`, `num_epochs`。
- 多语言：`i18n.py` 控制演示的语言包，Streamlit 页面会根据所选语言加载对应文本。

### 运行训练与评估（示例）

```powershell
python train.py --config config.yaml
python demo_run.py --input examples/sample.jpg --output gradcam.png
```

### 注意事项

- 若仓库无法直接推送到 GitHub（网络受限），请使用 `tools/network_fallback/` 中的 `repo.bundle` 或 `patches` 导出工具，在可联网机器上导入并推送（详见 `tools/network_fallback/README.md`）。
- 在执行历史重写并强制推送前，请备份并通知所有协作者。

## 更新日志（要点）

- 已加入 pre-commit（Black/isort/flake8）和 GitHub Actions CI。
- 添加了 `tools/network_fallback/` 用于在无法直接推送到 GitHub 时导出 `repo.bundle` 和补丁。

## 主要功能

- 训练与评估脚本（`train.py`）
- 轻量模型（ResNet-18）与检查点加载（`model.py`）
- Grad-CAM 可视化工具（`utils.py`）
- Streamlit 演示页面（`app.py`）
- 多语言支持与国际化（`i18n.py`）

## 贡献

请阅读 `CONTRIBUTING.md` 并在提交 PR 前运行：

```powershell
pre-commit run --all-files
```

## 许可

本项目采用 Apache-2.0 许可，详见 `LICENSE` 文件。
