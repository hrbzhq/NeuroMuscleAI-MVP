import os

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

from i18n import t
from model import build_model
from utils import grad_cam, overlay_heatmap
import numpy as np
import torchvision.transforms as T
import io
import zipfile
import csv

from inference import predict_with_probs, entropy_from_probs

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 语言选择（默认：中文）
langs = {"中文": "zh", "English": "en", "日本語": "jp"}
lang_choice = st.sidebar.selectbox("Language / 言語 / 语言", list(langs.keys()), index=0)
lang = langs[lang_choice]
lang = langs[lang_choice]

@st.cache_resource
def get_model(path="models/best_model.pth", pretrained=False):
    """Cached model loader. Returns a model already moved to `device`.
    Using a cached resource prevents Streamlit from reloading model on every rerun.
    """
    m = build_model(pretrained=pretrained)
    best_path = os.path.join(os.getcwd(), path)
    if os.path.exists(best_path):
        try:
            m.load_state_dict(torch.load(best_path, map_location=device))
            print("Loaded best model from", best_path)
        except Exception as e:
            print("Failed to load best model:", e)
    else:
        print(t(lang, "no_model"))
    return m.to(device)

# 侧边栏：模型元数据与快速操作
st.sidebar.markdown("### 模型信息")
st.sidebar.write(f"Device: {device}")
st.sidebar.write(f"Model file: {best_path if os.path.exists(best_path) else '（未找到）'}")
# 置信度/温度缩放设置
st.sidebar.markdown("### 预测设置")
temp = st.sidebar.slider("Temperature (softmax scale)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
confidence_threshold = st.sidebar.slider("Confidence threshold (人工复核阈值)", min_value=0.5, max_value=0.99, value=0.75, step=0.01)
if st.sidebar.button("快速演示（随机样本）"):
    # 生成随机合成图像并在主区显示预测与 Grad-CAM
    demo_img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    st.image(demo_img, caption="随机合成样本", use_container_width=True)
    transform_demo = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    demo_tensor = transform_demo(demo_img).unsqueeze(0).to(device)
    try:
        pred, probs = predict_with_probs(model, demo_tensor, temperature=temp)
        demo_result = t(lang, "normal") if pred == 0 else t(lang, "atrophy")
        st.subheader("随机样本预测")
        st.success(demo_result)
        st.write(f"概率: {np.round(probs,3).tolist()}")
        entropy = entropy_from_probs(probs)
        st.write(f"预测熵 (uncertainty): {entropy:.4f}")
        if probs.max() < confidence_threshold or entropy > 0.6:
            st.warning("置信度较低：建议人工复核此样本。")
        heatmap = grad_cam(model, demo_tensor)
        overlay = overlay_heatmap(demo_img, heatmap)
        st.image(overlay, caption="随机样本 Grad-CAM", use_container_width=True)
    except Exception as e:
        st.error(f"随机演示失败: {e}")

# 获取缓存的模型实例（已移动到 device）
model = get_model()

st.set_page_config(page_title="NeuroMuscle AI", layout="centered")
st.title(t(lang, "title"))
st.markdown(t(lang, "subtitle"))

uploaded_file = st.file_uploader(t(lang, "upload_prompt"), type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=t(lang, "upload_preview"), use_container_width=True)

    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    input_tensor = transform(image).unsqueeze(0).to(device)
    pred, probs = predict_with_probs(model, input_tensor, temperature=temp)
    result = t(lang, "normal") if pred == 0 else t(lang, "atrophy")

    st.subheader(t(lang, "predict_result"))
    st.success(result)
    st.write(f"概率: {np.round(probs,3).tolist()}")
    entropy = entropy_from_probs(probs)
    st.write(f"预测熵 (uncertainty): {entropy:.4f}")
    if probs.max() < confidence_threshold or entropy > 0.6:
        st.warning("置信度较低：建议人工复核此样本。")

    overlay_saved = None
    if st.button(t(lang, "generate_gradcam")):
        try:
            heatmap = grad_cam(model, input_tensor)
            overlay = overlay_heatmap(image, heatmap)
            st.image(overlay, caption=t(lang, "gradcam_caption"), use_container_width=True)
            # 保存示例文件到磁盘
            overlay.save("sample_overlay.jpg")
            overlay_saved = "sample_overlay.jpg"
            st.write(t(lang, "saved_overlay"), overlay_saved)
        except Exception as e:
            st.error(f'{t(lang, "generate_gradcam")} failed: {e}')

    if overlay_saved:
        with open(overlay_saved, "rb") as f:
            st.download_button(label=t(lang, "download_overlay"), data=f, file_name="sample_overlay.jpg")

# Batch evaluation UI
batch_mode = st.sidebar.checkbox("Batch evaluation", value=False)
if batch_mode:
    st.header("Batch evaluation")
    st.write("Upload multiple images (or a zip) and an optional CSV with filename,label to run batch evaluation.")
    uploaded_files = st.file_uploader("Images or zip", accept_multiple_files=True, type=["png", "jpg", "jpeg", "zip"]) 
    labels_csv = st.file_uploader("Labels CSV (filename,label)", type=["csv"]) 
    max_samples = st.number_input("Max samples to process", min_value=1, max_value=500, value=64)
    if labels_csv is not None:
        try:
            s = labels_csv.read().decode('utf-8')
            reader = csv.reader(s.splitlines())
            label_map = {row[0].strip(): int(row[1].strip()) for row in reader if len(row) >= 2}
        except Exception as e:
            st.warning(f"Failed to parse CSV: {e}")
            label_map = {}
    else:
        label_map = {}

    # collect image entries (filename, bytes)
    image_entries = []
    if uploaded_files:
        for f in uploaded_files:
            name = f.name
            data = f.read()
            if name.lower().endswith('.zip'):
                try:
                    z = zipfile.ZipFile(io.BytesIO(data))
                    for zi in z.infolist():
                        if zi.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_entries.append((zi.filename, z.read(zi)))
                except Exception as e:
                    st.warning(f"Failed to read zip {name}: {e}")
            else:
                image_entries.append((name, data))

    if st.button("Run batch evaluation") and len(image_entries) > 0:
        st.info(f"Processing {min(len(image_entries), max_samples)} samples...")
        preds = []
        truths = []
        filenames = []
        for i, (fname, buf) in enumerate(image_entries[:max_samples]):
            try:
                img = Image.open(io.BytesIO(buf)).convert('RGB')
                transform = T.Compose([T.Resize((224,224)), T.ToTensor()])
                inp = transform(img).unsqueeze(0).to(device)
                pred_label, probs = predict_with_probs(model, inp, temperature=temp)
                preds.append(int(pred_label))
                truths.append(label_map.get(fname, None))
                filenames.append(fname)
            except Exception as e:
                st.warning(f"{fname}: {e}")

        has_truth = any(v is not None for v in truths)
        if has_truth:
            classes = sorted(list(set([p for p in preds] + [t for t in truths if t is not None])))
            cls_to_idx = {c: i for i, c in enumerate(classes)}
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for gt, pr in zip(truths, preds):
                if gt is None:
                    continue
                cm[cls_to_idx[gt], cls_to_idx[pr]] += 1

            # per-class metrics
            rows = []
            for i_c, c in enumerate(classes):
                tp = cm[i_c, i_c]
                fp = cm[:, i_c].sum() - tp
                fn = cm[i_c, :].sum() - tp
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                rows.append({'class': c, 'precision': round(prec,3), 'recall': round(rec,3), 'f1': round(f1,3)})

            st.subheader('Confusion matrix')
            st.write(cm)
            st.subheader('Per-class metrics')
            st.table(rows)

        # show sample predictions
        display = [{'file': fn, 'pred': pr, 'truth': (gt if gt is not None else '-') } for fn,pr,gt in zip(filenames, preds, truths)]
        st.subheader('Sample predictions')
        st.table(display[:max_samples])

        # show grad-cam samples if available
        try:
            from utils import grad_cam, overlay_heatmap
            st.subheader('Grad-CAM samples')
            show_n = min(6, len(filenames))
            cols = st.columns(min(3, show_n))
            for i in range(show_n):
                fn = filenames[i]
                buf = image_entries[i][1]
                img = Image.open(io.BytesIO(buf)).convert('RGB')
                heat = grad_cam(model, transform(img).unsqueeze(0).to(device))
                overlay = overlay_heatmap(img, heat)
                with cols[i % len(cols)]:
                    st.image(overlay, caption=f"{fn} -> {preds[i]}")
        except Exception:
            pass

with st.expander(f"📖 {t(lang, 'background')}"):
    st.markdown(t(lang, "subtitle"))

st.markdown("---")
st.caption(t(lang, "footer"))
