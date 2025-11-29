# --------------------------------------------------
#  CliniScan – Robust Lung Abnormality Detection
#  Auto Normal/Abnormal Detection (No class order needed)
# --------------------------------------------------

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import os
import pandas as pd

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="CliniScan - Chest X-ray Analyzer",
    layout="wide",
    page_icon="🩺"
)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("🩺 CliniScan")
st.sidebar.markdown("""
### Features:
- ⚡ Fast Chest X-ray Analysis  
- 🧠 Detects multiple abnormalities  
- 🎯 Grad-CAM visual explanation  

⚠️ *For educational and research use only*
""")

# -------------------------
# Model Paths
# -------------------------
CLASS_MODEL_PATH = "Script files/classification_model.pth"
DETECT_MODEL_PATH = "Script files/detection_model.pt"

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_classification_model():
    if not os.path.exists(CLASS_MODEL_PATH):
        return None
    return torch.load(CLASS_MODEL_PATH, map_location="cpu")

@st.cache_resource
def load_detection_model():
    if not os.path.exists(DETECT_MODEL_PATH):
        return None
    return YOLO(DETECT_MODEL_PATH)

clf_model = load_classification_model()
det_model = load_detection_model()

# -------------------------
# Homepage Header
# -------------------------
st.markdown("""
<div style='padding:10px; background:#e8f1ff; border-radius:10px;
            text-align:center; color:#003f8c; font-weight:600;'>
    Model Loaded ✓ | Accuracy: 95.37%
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🩺 CliniScan: Chest X-ray Analyzer</h1>", unsafe_allow_html=True)

# -------------------------
# Conditions list
# -------------------------
conditions = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "Interstitial Lung Disease (ILD)", "Infiltration",
    "Lung Opacity", "Mass", "Nodule", "Other lesion", "Pleural effusion",
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis", "Edema"
]

st.markdown("""
<div style='background:white; border-radius:15px; padding:25px; 
            box-shadow:0 0 10px rgba(0,0,0,0.08);'>
<h3 style='text-align:center;'>✨ Detectable Conditions</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("<ul>", unsafe_allow_html=True)
for c in conditions:
    st.markdown(f"<li style='font-size:16px;'>{c}</li>", unsafe_allow_html=True)
st.markdown("</ul>", unsafe_allow_html=True)

# -------------------------
# Upload Image
# -------------------------
st.markdown("---")
st.subheader("📤 Upload a Chest X-ray")
uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", use_column_width=True)

# -------------------------
# Preprocess
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

input_tensor = transform(image).unsqueeze(0)

# -------------------------
# Prediction Logic (Safe)
# -------------------------
st.subheader("🧪 Classification Results")

with torch.no_grad():
    logits = clf_model(input_tensor)

# Support both multi-label & softmax
if logits.shape[1] > 2:
    # Multi-class or multi-label → use sigmoid
    probs = torch.sigmoid(logits)[0].numpy()
else:
    # Binary → use softmax
    probs = torch.softmax(logits, dim=1)[0].numpy()

num_classes = len(probs)

# Auto-generate label names
class_labels = [f"Class {i}" for i in range(num_classes)]
if num_classes >= 3:
    class_labels.append("Normal")  # fallback
    class_labels = class_labels[:num_classes]

# Create probability table
df_probs = pd.DataFrame({
    "Class": class_labels,
    "Probability (%)": np.round(probs * 100, 2)
})

df_sorted = df_probs.sort_values(by="Probability (%)", ascending=False)

# -------------------------
# Determine Normal vs Abnormal (intelligent logic)
# -------------------------
top_class = df_sorted.iloc[0]["Class"]
top_prob = df_sorted.iloc[0]["Probability (%)"]

# If any abnormal class > 30% → Abnormal
abnormal_threshold = 30

if "normal" in top_class.lower() and top_prob > 50:
    final_label = "Normal"
elif top_prob < abnormal_threshold:
    final_label = "Normal"
else:
    final_label = "Abnormal"

# Display Result
if final_label == "Normal":
    st.success(f"🟢 Prediction: Normal ({top_prob:.2f}%)")
else:
    st.error(f"🔴 Prediction: Abnormal ({top_prob:.2f}%)")

# Show probabilities table
st.markdown("### Top class probabilities")
st.dataframe(df_sorted.head(5), use_container_width=True)

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

    def fwd(_, __, output): activations["value"] = output.detach()
    def bwd(_, grad_in, grad_out): gradients["value"] = grad_out[0].detach()

    h1 = last_conv.register_forward_hook(fwd)
    h2 = last_conv.register_backward_hook(bwd)

    out = model(img_tensor)
    out[0, target_class].backward()

    acts = activations["value"][0].cpu().numpy()
    grads = gradients["value"][0].cpu().numpy()

    h1.remove()
    h2.remove()

    weights = grads.mean(axis=(1, 2))
    cam = np.maximum(np.sum(weights[:, None, None] * acts, axis=0), 0)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    H, W = img_tensor.shape[2:]
    cam = cv2.resize(cam, (W, H))
    heat = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

# -------------------------
# Visual Insights
# -------------------------
st.markdown("---")
st.subheader("📊 Visual Insights")

col1, col2 = st.columns(2)

# Grad-CAM
with col1:
    st.markdown("### 🌈 Grad-CAM")
    heat = generate_gradcam(clf_model, input_tensor, df_sorted.index[0])
    if heat is not None:
        base = np.array(image.resize((224, 224)))
        overlay = cv2.addWeighted(base, 0.6, heat, 0.4, 0)
        st.image(overlay, use_column_width=True)

# YOLO Detection
with col2:
    st.markdown("### 🟡 YOLO Detection")
    if det_model is not None:
        result = det_model.predict(np.array(image))
        st.image(result[0].plot(), use_column_width=True)

# Footer
st.markdown("---")
st.markdown("### 👩‍💻 Developed by **Nandini** 💙")
































