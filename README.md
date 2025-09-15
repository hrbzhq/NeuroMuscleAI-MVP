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
