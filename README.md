# NeuroMuscleAI-MVP

AI 识别肌肉衰退图像的开源原型 · Beginnings 生态系统

发起人: beginningstone

## 目标
构建一个可运行的 MVP，用于对肌肉组织图像进行“正常 / 衰退”二分类，并提供可视化与训练日志记录。项目面向高校与科研院所，支持本地部署与扩展。

## 快速开始
1. 创建并激活 Python 环境（建议 Python 3.8+）

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## 文件结构
- `app.py` - Streamlit 前端原型
- `model.py` - 模型构建与加载
- `train.py` - 训练脚本
- `utils.py` - 工具函数（预处理、可视化、Grad-CAM 等）
- `config.yaml` - 实验配置
- `requirements.txt` - 依赖

## 许可证
MIT License

## 示例输出
下面展示从一次短跑 k-fold 演示中保存的 Grad-CAM 叠加样例（位于 `logs/` 目录）。这些图像用于展示模型关注区域，便于开源演示与结果复现。

示例图（来自 `logs/`）：

![Fold1 Grad-CAM](logs/fold1_gradcam_e1_s0.jpg)
![Fold2 Grad-CAM](logs/fold2_gradcam_e1_s0.jpg)
![Fold3 Grad-CAM](logs/fold3_gradcam_e1_s0.jpg)

如何复现短跑（在项目根目录运行）：

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --config config.yaml --k_folds 3 --epochs 2
```

说明：运行结束后，模型权重将保存在 `models/`，日志与 Grad-CAM 样例存放在 `logs/`，分类报告为 `logs/fold{n}_report_epoch{e}.txt`。

## Demo 指南
下面是逐步演示如何在本地重现实验（包括 k-fold 演示、可选的 wandb 接入，以及如何在 Streamlit 前端加载并展示 `best_model.pth`）。所有命令为 PowerShell 风格，适用于 Windows。

1) 环境准备

```powershell
# 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

2) 运行 k-fold 演示（短跑示例）

```powershell
# 在项目根运行一个小规模示例（例如 k=3, epochs=2）用于快速验证
python train.py --config config.yaml --k_folds 3 --epochs 2

# 运行结束后检查产物：
dir models
dir logs
```

说明：训练脚本会把每折的最佳权重保存在 `models/best_model_fold{n}.pth`，并在最后保存 `models/model_final.pth`。日志（TensorBoard）保存在 `logs/`。

3) 启用 Weights & Biases（可选）

1. 注册并获取 API Key：到 https://wandb.ai 注册并在 Settings 中拷贝 API Key。
2. 在本地登录（仅需一次）：

```powershell
wandb login <YOUR_API_KEY>
```

3. 在 `config.yaml` 中把 `use_wandb` 设置为 `True`：

```yaml
use_wandb: True
experiment_name: "muscle_atrophy_v1"
```

4. 重新运行 `train.py`，训练过程中的指标与 Grad-CAM 图片将被上报到 wandb（如果网络与登录正常）。

注意：如果不希望在公共项目中暴露 API Key，请使用本地 CI secret 或系统环境变量。

4) 在 Streamlit 中加载 `best_model.pth`

`app.py` 在启动时会尝试加载 `models/best_model.pth` 或 `models/model_final.pth`（如果存在）。要在 Streamlit 中加载并演示：

```powershell
# 启动 Streamlit 前端
streamlit run app.py

# 在网页中上传图像，点击 Predict/Grad-CAM 按钮即可查看预测与热力图叠加。
```

如果您想指定某个特定的权重文件，可以在 `app.py` 中修改加载路径或在 UI 中添加一个文件选择逻辑（目前程序会优先读取 `models/best_model.pth`）。

5) 常见故障与调试提示
- 如果训练时报 `ModuleNotFoundError` 或缺少包，请确保虚拟环境已激活并运行 `pip install -r requirements.txt`。
- 如果 wandb 上传失败，请检查网络与 `wandb login` 状态；也可以暂时把 `use_wandb` 设为 `False`。
- Grad-CAM 报错通常与模型结构（没有 Conv2d 层）或输入尺寸不符有关；确保使用 `model.build_model(pretrained=True)` 的默认 ResNet18 或调整 `utils.grad_cam` 的层选择逻辑。

欢迎把这个项目 Fork 到 GitHub 并在 Issues 中提交问题或改进建议。我们建议在推送到远端前确认 `.gitignore` 已正确忽略 `models/`, `logs/`, `data/` 等大文件目录。

发布指南：请参考 `RELEASE.md`，其中包含如何打 tag、创建 Release、以及如何安全地处理模型权重和大文件的建议。
