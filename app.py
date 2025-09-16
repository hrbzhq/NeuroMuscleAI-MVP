import os

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

from i18n import t
from model import build_model
from utils import grad_cam, overlay_heatmap

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 语言选择（默认：中文）
langs = {"中文": "zh", "English": "en", "日本語": "jp"}
lang_choice = st.sidebar.selectbox("Language / 言語 / 语言", list(langs.keys()), index=0)
lang = langs[lang_choice]

# 尝试加载最佳模型（如果存在）
model = build_model(pretrained=False)
best_path = os.path.join("models", "best_model.pth")
if os.path.exists(best_path):
    try:
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("Loaded best model from", best_path)
    except Exception as e:
        print("Failed to load best model:", e)
else:
    print(t(lang, "no_model"))
model = model.to(device)

st.set_page_config(page_title="NeuroMuscle AI", layout="centered")
st.title(t(lang, "title"))
st.markdown(t(lang, "subtitle"))

uploaded_file = st.file_uploader(t(lang, "upload_prompt"), type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=t(lang, "upload_preview"), use_column_width=True)

    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        result = t(lang, "normal") if predicted.item() == 0 else t(lang, "atrophy")

    st.subheader(t(lang, "predict_result"))
    st.success(result)

    overlay_saved = None
    if st.button(t(lang, "generate_gradcam")):
        try:
            heatmap = grad_cam(model, input_tensor)
            overlay = overlay_heatmap(image, heatmap)
            st.image(overlay, caption=t(lang, "gradcam_caption"), use_column_width=True)
            # 保存示例文件到磁盘
            overlay.save("sample_overlay.jpg")
            overlay_saved = "sample_overlay.jpg"
            st.write(t(lang, "saved_overlay"), overlay_saved)
        except Exception as e:
            st.error(f'{t(lang, "generate_gradcam")} failed: {e}')

    if overlay_saved:
        with open(overlay_saved, "rb") as f:
            st.download_button(label=t(lang, "download_overlay"), data=f, file_name="sample_overlay.jpg")

with st.expander(f"📖 {t(lang, 'background')}"):
    st.markdown(t(lang, "subtitle"))

st.markdown("---")
st.caption(t(lang, "footer"))
