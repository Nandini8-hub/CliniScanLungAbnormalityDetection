# -------------------------
#  CliniScan – Lung Abnormality Detection  
#  UPDATED FULL app.py WITH LOGO + THEME + UI IMPROVEMENTS
# -------------------------

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.models.efficientnet import EfficientNet
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import os

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="CliniScan - Lung Abnormality Detection",
    layout="wide",
    page_icon="🩺"
)

# -------------------------
# CUSTOM CSS THEME (like screenshot)
# -------------------------
st.markdown("""
<style>

body {
    background-color: #ffffff;
}

.sidebar .sidebar-content {
    background-color: #f8fafc;
}

.logo {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 140px;
    border-radius: 12px;
    margin-bottom: 15px;
}

.header-banner {
    padding: 12px;
    background: #e7f1ff;
    border-radius: 12px;
    color: #003d99;
    text-align: center;
    font-weight: 600;
    font-size: 17px;
}

.condition-box {
    background: #ffffff;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 0 8px rgba(0,0,0,0.06);
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# LOGO + SIDEBAR
# -------------------------
st.sidebar.markdown(
    "<img src='https://i.imgur.com/E66QGZo.png' class='logo'>",
    unsafe_allow_html=True
)

st.sidebar.title("🩺CliniScan")
st.sidebar.markdown("""
### Features:
- ⚡ **Fast Chest X-ray Analysis**
- 🧠 Detects multiple abnormalities  
- 🎯 Visual explanations (Grad-CAM)

### Classes:  
**Normal**, **Abnormal**

⚠️ *For educational, research & non-clinical use only!*
""")

# -------------------------
# PATHS
# -------------------------
CLASS_MODEL_PATH = os.path.join("Script files", "classification_model.pth")
DETECT_MODEL_PATH = os.path.join("Script files", "detection_model.pt")

# Supported classes (your model)
CLASS_NAMES = [
    "Normal", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_classification_model():
    if not os.path.exists(CLASS_MODEL_PATH):
        return None
    try:
        torch.serialization.add_safe_globals([EfficientNet])
    except:
        pass
    return torch.load(CLASS_MODEL_PATH, map_location="cpu", weights_only=False)

@st.cache_resource
def load_detection_model():
    if not os.path.exists(DETECT_MODEL_PATH):
        return None
    return YOLO(DETECT_MODEL_PATH)

clf_model = load_classification_model()
det_model = load_detection_model()

# -------------------------
# HEADER BANNER
# -------------------------
st.markdown("""
<div class='header-banner'>
    Model Loaded | Accuracy: <b>92.58%</b>
</div>
""", unsafe_allow_html=True)

# -------------------------
# MAIN TITLE
# -------------------------
st.markdown("""
<h1 style='text-align:center; margin-top:10px;'>🩺 CliniScan – Lung Abnormality Detection</h1>
""", unsafe_allow_html=True)

# -------------------------
# DETECTABLE CONDITIONS CARD
# -------------------------
st.markdown("<div class='condition-box'>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>✨ Detectable Conditions</h3>", unsafe_allow_html=True)

conditions = [
    "Aortic enlargement", "Atelectasis", "Calcification",
    "Cardiomegaly", "Consolidation", "Interstitial Lung Disease (ILD)",
    "Infiltration", "Lung Opacity", "Nodule / Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis", "Edema"
]

st.markdown("<ul>", unsafe_allow_html=True)
for c in conditions:
    st.markdown(f"<li style='font-size:17px;'>{c}</li>", unsafe_allow_html=True)
st.markdown("</ul></div>", unsafe_allow_html=True)

# -------------------------
# IMAGE UPLOAD
# -------------------------
st.markdown("---")
st.subheader("📤 Upload a Chest X-ray Image")

file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
if file is None:
    st.stop()

image = Image.open(file).convert("RGB")
st.image(image, caption="Uploaded Image", use_column_width=True)

# -------------------------
# PREPROCESSING
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)

# -------------------------
# CLASSIFICATION
# -------------------------
st.subheader("🔍 Classification Results")

with torch.no_grad():
    logits = clf_model(input_tensor)
    probs = torch.softmax(logits, dim=1)[0].numpy()

idx = int(np.argmax(probs))
label = CLASS_NAMES[idx]
confidence = probs[idx] * 100

st.markdown(f"### **Prediction:** {label} — {confidence:.2f}%")

if label == "Normal" and confidence > 50:
    st.success("🟢 NORMAL")
else:
    st.error("🔴 ABNORMAL")

# -------------------------
# GRAD-CAM FUNCTION
# -------------------------
def generate_gradcam(model, img_tensor, target_class):
    activations = {}
    gradients = {}
    last_conv = None

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m

    def fwd_hook(_, __, output):
        activations["value"] = output.detach()

    def bwd_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    fh = last_conv.register_forward_hook(fwd_hook)
    bh = last_conv.register_backward_hook(bwd_hook)

    out = model(img_tensor)
    out[0, target_class].backward()

    acts = activations["value"][0].numpy()
    grads = gradients["value"][0].numpy()

    fh.remove()
    bh.remove()

    weights = grads.mean(axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    H, W = img_tensor.shape[2:]
    cam = cv2.resize(cam, (W, H))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# -------------------------
# SIDE-BY-SIDE VISUAL RESULTS
# -------------------------
st.markdown("---")
st.subheader("📊 Visual Insights")

col1, col2 = st.columns(2)

# LEFT → GradCAM
with col1:
    st.markdown("### 🌈 Grad-CAM Visualization")
    heatmap = generate_gradcam(clf_model, input_tensor, idx)
    img_small = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(img_small, 0.6, heatmap, 0.4, 0)
    st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)

# RIGHT → YOLO Detection
with col2:
    st.markdown("### 🟡 YOLO Detection")
    results = det_model.predict(np.array(image))
    st.image(results[0].plot(), caption="Detected Abnormal Regions", use_column_width=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("""
### 👩‍💻 Developed by **Nandini 💙**  
#### For research, academic & educational use.
""")



























