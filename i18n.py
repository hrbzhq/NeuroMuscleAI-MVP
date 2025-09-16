translations = {
    "zh": {
        "title": "🧠 NeuroMuscle AI - 肌肉衰退识别原型",
        "subtitle": "本项目由 **beginningstone** 发起，服务于再生医学与AI融合的科研探索。",
        "upload_prompt": "请上传肌肉组织图像（JPG/PNG）",
        "upload_preview": "上传图像预览",
        "predict_result": "预测结果",
        "normal": "🟢 正常肌肉组织",
        "atrophy": "🔴 肌肉衰退特征",
        "generate_gradcam": "生成 Grad-CAM 热力图",
        "gradcam_caption": "Grad-CAM 叠加热力图",
        "saved_overlay": "已生成并保存：",
        "download_overlay": "下载叠加图像",
        "no_model": "未找到模型，正在使用随机初始化模型进行演示",
        "background": "项目背景与愿景",
        "footer": "© 2025 beginningstone · NeuroMuscleAI-MVP · MIT License",
    },
    "en": {
        "title": "🧠 NeuroMuscle AI - Muscle Atrophy Detection (Prototype)",
        "subtitle": "This project is initiated by **beginningstone** to support regenerative medicine research with AI tools.",
        "upload_prompt": "Upload a muscle tissue image (JPG/PNG)",
        "upload_preview": "Uploaded image preview",
        "predict_result": "Prediction",
        "normal": "🟢 Normal muscle tissue",
        "atrophy": "🔴 Signs of muscle atrophy",
        "generate_gradcam": "Generate Grad-CAM heatmap",
        "gradcam_caption": "Grad-CAM overlay",
        "saved_overlay": "Generated and saved:",
        "download_overlay": "Download overlay image",
        "no_model": "No model found; using randomly initialized model for demo",
        "background": "Project background & vision",
        "footer": "© 2025 beginningstone · NeuroMuscleAI-MVP · MIT License",
    },
    "jp": {
        "title": "🧠 NeuroMuscle AI - 筋萎縮検出プロトタイプ",
        "subtitle": "このプロジェクトは **beginningstone** により開始され、再生医療研究をAIで支援します。",
        "upload_prompt": "筋組織の画像をアップロードしてください（JPG/PNG）",
        "upload_preview": "アップロード画像プレビュー",
        "predict_result": "予測結果",
        "normal": "🟢 正常な筋繊維",
        "atrophy": "🔴 筋萎縮の兆候",
        "generate_gradcam": "Grad-CAM ヒートマップを生成",
        "gradcam_caption": "Grad-CAM オーバーレイ",
        "saved_overlay": "生成して保存しました：",
        "download_overlay": "オーバーレイ画像をダウンロード",
        "no_model": "モデルが見つかりません。デモのためランダム初期化モデルを使用します",
        "background": "プロジェクトの背景とビジョン",
        "footer": "© 2025 beginningstone · NeuroMuscleAI-MVP · MIT License",
    },
}


def t(lang, key):
    """Get translated string for given language and key. Falls back to English then key."""
    if lang in translations and key in translations[lang]:
        return translations[lang][key]
    if "en" in translations and key in translations["en"]:
        return translations["en"][key]
    return key
