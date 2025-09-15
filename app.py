import streamlit as st
from PIL import Image
import torch
from model import build_model
import torchvision.transforms as transforms
from utils import grad_cam, overlay_heatmap, load_image
import os

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 尝试加载最佳模型（如果存在）
model = build_model(pretrained=False)
best_path = os.path.join('models', 'best_model.pth')
if os.path.exists(best_path):
    try:
        model.load_state_dict(torch.load(best_path, map_location=device))
        print('Loaded best model from', best_path)
    except Exception as e:
        print('Failed to load best model:', e)
else:
    print('No best model found at models/best_model.pth; using randomly initialized model')
model = model.to(device)

st.set_page_config(page_title="NeuroMuscle AI", layout="centered")
st.title("🧠 NeuroMuscle AI - 肌肉衰退识别原型")
st.markdown("本项目由 **beginningstone** 发起，服务于再生医学与AI融合的科研探索。")

uploaded_file = st.file_uploader("请上传肌肉组织图像（JPG/PNG）", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="上传图像预览", use_column_width=True)

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        result = "🟢 正常肌肉组织" if predicted.item() == 0 else "🔴 肌肉衰退特征"

    st.subheader("预测结果")
    st.success(result)

    overlay_saved = None
    if st.button('生成 Grad-CAM 热力图'):
        try:
            heatmap = grad_cam(model, input_tensor)
            overlay = overlay_heatmap(image, heatmap)
            st.image(overlay, caption='Grad-CAM 叠加热力图', use_column_width=True)
            # 保存示例文件到磁盘
            overlay.save('sample_overlay.jpg')
            overlay_saved = 'sample_overlay.jpg'
            st.write('已生成并保存：sample_overlay.jpg')
        except Exception as e:
            st.error(f'生成 Grad-CAM 失败: {e}')

    if overlay_saved and st.button('下载叠加图像'):
        with open(overlay_saved, 'rb') as f:
            btn = st.download_button(label='下载 overlay', data=f, file_name='sample_overlay.jpg')

    if st.button('生成 Grad-CAM 热力图'):
        try:
            heatmap = grad_cam(model, input_tensor)
            overlay = overlay_heatmap(image, heatmap)
            st.image(overlay, caption='Grad-CAM 叠加热力图', use_column_width=True)
            # 保存示例文件到磁盘
            overlay.save('sample_overlay.jpg')
            st.write('已生成并保存：sample_overlay.jpg')
        except Exception as e:
            st.error(f'生成 Grad-CAM 失败: {e}')

with st.expander("📖 项目背景与愿景"):
    st.markdown("""
    NeuroMuscle AI 是 “Beginnings生态系统” 的一部分，致力于通过AI辅助识别肌肉衰退图像，
    为再生医学研究提供智能工具。未来将拓展至细胞重编程、药物筛选等模块。
    """)

st.markdown("---")
st.caption("© 2025 beginningstone · NeuroMuscleAI-MVP · MIT License")
