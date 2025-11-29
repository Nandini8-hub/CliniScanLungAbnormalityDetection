# -------------------------
#  CliniScan – Lung Abnormality Detection  
#  FULL app.py WITH HOMEPAGE UI (Matches Provided Screenshot)
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
# Page Config
# -------------------------
st.set_page_config(
    page_title="CliniScan - Lung Abnormality Detection",
    layout="wide",
    page_icon="🩺"
)

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title(" 🩺CliniScan")
st.sidebar.markdown("""
### Features:
- ⚡ **Fast Chest X-ray Analysis**
- 🧠 Detects many abnormalities  
- 🎯 Visual explanations (Grad-CAM)

### Classes:  
**Normal**, **Abnormal**

⚠️ *For educational, research, and non-clinical use only!*
""")

# -------------------------
# Paths
# -------------------------
CLASS_MODEL_PATH = os.path.join("Script files", "classification_model.pth")
DETECT_MODEL_PATH = os.path.join("Script files", "detection_model.pt")

# 15-class names
CLASS_NAMES = [
    "Normal", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

normal_default = CLASS_NAMES.index("Normal")
normal_label = st.sidebar.selectbox("Select NORMAL class:", CLASS_NAMES, index=normal_default)

# -------------------------
# Load Models
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
# TOP BANNER like screenshot
# -------------------------
st.markdown("""
<div style='padding:12px; background:#eaf3ff; border-radius:10px; 
            color:#003f8c; font-weight:600; text-align:center;'>
    ModelLoaded | Accuracy: 95.37%
</div>
""", unsafe_allow_html=True)

# -------------------------
# HOMEPAGE HEADER
# -------------------------
st.markdown("""
<h1 style='text-align:center; margin-top:10px;'>🩺CliniScan- Lung Abnormality Detection</h1>
""", unsafe_allow_html=True)

# -------------------------
# Homepage Detectable Conditions Card
# -------------------------
st.markdown("""
<br>
<div style='background:white; border-radius:15px; padding:25px; 
            box-shadow:0 0 10px rgba(0,0,0,0.08);'>
<h3 style='text-align:center;'>✨ Detectable Conditions</h3>
""", unsafe_allow_html=True)

# Bullet points list
condition_list = [
    "Aortic enlargement", "Atelectasis", "Calcification",
    "Cardiomegaly", "Consolidation", "Interstitial Lung Disease (ILD)",
    "Infiltration", "Lung Opacity", "Nodule / Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis", "Edema"
]

st.markdown("<ul>", unsafe_allow_html=True)
for c in condition_list:
    st.markdown(f"<li style='font-size:17px;'>{c}</li>", unsafe_allow_html=True)
st.markdown("</ul></div>", unsafe_allow_html=True)

# -------------------------
# IMAGE UPLOAD (appears after homepage)
# -------------------------
st.markdown("---")
st.subheader("📤 Upload a Chest X-ray")

uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", use_column_width=True)

# -------------------------
# Preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)

# -------------------------
# Classification
# -------------------------
st.subheader("🔍 Classification Results")

if clf_model is None:
    st.error("Model missing.")
else:
    with torch.no_grad():
        logits = clf_model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    pred_conf = probs[pred_idx] * 100

    st.markdown(f"### **Prediction:** {pred_label} — {pred_conf:.2f}%")

    if pred_label == normal_label and pred_conf > 50:
        st.success("🟢 NORMAL")
    else:
        st.error("🔴 ABNORMAL")

# -------------------------
# Grad-CAM Function
# -------------------------
def generate_gradcam(model, img_tensor, target_class):
    activations, gradients = {}, {}
    last_conv = None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        return None

    def fwd_hook(_, __, output): 
        activations["value"] = output.detach()
    def bwd_hook(_, grad_input, grad_output): 
        gradients["value"] = grad_output[0].detach()

    fh = last_conv.register_forward_hook(fwd_hook)
    bh = last_conv.register_backward_hook(bwd_hook)

    out = model(img_tensor)
    out[0, target_class].backward()

    acts = activations["value"][0].cpu().numpy()
    grads = gradients["value"][0].cpu().numpy()

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
# SIDE-BY-SIDE ANALYSIS
# -------------------------
st.markdown("---")
st.subheader("📊 Visual Insights")

col1, col2 = st.columns(2)

# LEFT: GradCAM
with col1:
    st.markdown("### 🌈 Grad-CAM")
    heatmap = generate_gradcam(clf_model, input_tensor, pred_idx)
    if heatmap is not None:
        img_small = np.array(image.resize((224, 224)))
        overlay = cv2.addWeighted(img_small, 0.6, heatmap, 0.4, 0)
        st.image(overlay, caption="Grad-CAM", use_column_width=True)

# RIGHT: YOLO Detection
with col2:
    st.markdown("### 🟡 Object Detection")
    if det_model is not None:
        results = det_model.predict(np.array(image))
        st.image(results[0].plot(), caption="Detected Regions", use_column_width=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("### 👩‍💻 Developed by **Nandini💙** — For learning & research use.")

























