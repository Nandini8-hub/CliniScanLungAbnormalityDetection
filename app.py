# -------------------------------------------------------------
# CliniScan – Lung Abnormality Detection
# Full app.py — Final Cleaned Version (Normal + Abnormal Working)
# -------------------------------------------------------------

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

# -------------------------------------------------------------
# Page Config
# -------------------------------------------------------------
st.set_page_config(
    page_title="CliniScan - Lung Abnormality Detection",
    layout="wide",
    page_icon="🩺"
)

# -------------------------------------------------------------
# CSS (Theme + Table Style + Logo Styling)
# -------------------------------------------------------------
st.markdown("""
<style>
.logo { width:150px; display:block; margin:auto; border-radius:12px; }
.header-banner { padding:12px; background:#eaf3ff; border-radius:10px; color:#003f8c; text-align:center; font-weight:600; }
.condition-box { background:#fff; border-radius:12px; padding:18px; box-shadow:0 0 8px rgba(0,0,0,0.06); }
table td, table th { padding:8px 14px; font-size:16px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# File Paths
# -------------------------------------------------------------
CLASS_MODEL_PATH = os.path.join("Script files", "classification_model.pth")
DETECT_MODEL_PATH = os.path.join("Script files", "detection_model.pt")

# -------------------------------------------------------------
# CLASSES
# -------------------------------------------------------------
CLASS_NAMES = [
    "Normal", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# -------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------
with st.sidebar:
    if os.path.exists("streamlit_logo.png"):
        st.image("streamlit_logo.png", use_column_width=False)
    else:
        st.image("https://i.imgur.com/E66QGZo.png", use_column_width=False)

    st.title("🩺 CliniScan")
    st.markdown("""
    **Features**
    - Fast Chest X-ray classification  
    - Grad-CAM visualizations  
    - YOLO Detection  

    ⚠ For **educational & research use only**
    """)

    st.markdown("---")
    st.subheader("Prediction Settings")
    normal_threshold = st.slider(
        "Normal threshold (%)",
        0.0, 1.0, 0.50, 0.01
    )

# -------------------------------------------------------------
# Load Models
# -------------------------------------------------------------
@st.cache_resource
def load_classification_model():
    try:
        torch.serialization.add_safe_globals([EfficientNet])
    except:
        pass

    model = torch.load(CLASS_MODEL_PATH, map_location="cpu")
    model.eval()
    return model


@st.cache_resource
def load_detection_model():
    return YOLO(DETECT_MODEL_PATH)


clf_model = load_classification_model()
det_model = load_detection_model()

# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
st.markdown("<div class='header-banner'>Model Loaded | Accuracy: <b>95.37%</b></div>",
            unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;'>🩺 CliniScan — Lung Abnormality Detection</h1>",
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# Detectable Conditions
# -------------------------------------------------------------
st.markdown("<div class='condition-box'>", unsafe_allow_html=True)
st.markdown("### ✨ Detectable Conditions", unsafe_allow_html=True)

cond = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Opacity", "Mass / Nodule",
    "Lesions", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Fibrosis", "Edema"
]

st.markdown("<ul>", unsafe_allow_html=True)
for c in cond:
    st.markdown(f"<li style='font-size:16px'>{c}</li>", unsafe_allow_html=True)
st.markdown("</ul></div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------------
# Upload Image
# -------------------------------------------------------------
st.subheader("📤 Upload a Chest X-ray Image")
upload = st.file_uploader("Choose an image", ["png", "jpg", "jpeg"])

if not upload:
    st.info("Upload an image to continue.")
    st.stop()

image = Image.open(upload).convert("RGB")
st.image(image, use_column_width=True)

# -------------------------------------------------------------
# Preprocess
# -------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

tensor = transform(image).unsqueeze(0)

# -------------------------------------------------------------
# Classification
# -------------------------------------------------------------
st.subheader("🔍 Classification Results")

with torch.no_grad():
    logits = clf_model(tensor)
    logits = logits[0] if logits.ndim > 1 else logits
    probs = torch.softmax(logits, dim=0).cpu().numpy()

normal_prob = float(probs[0])
abnormal_prob = float(1 - normal_prob)

# -----------------------------
# YOUR CUSTOM RULE ADDED HERE
# -----------------------------
label = "Normal" if normal_prob >= abnormal_prob else "Abnormal"

confidence = normal_prob * 100 if label == "Normal" else abnormal_prob * 100

if label == "Normal" and confidence > 50:
    st.success(f"🟢 NORMAL ({confidence:.2f}%)")
else:
    st.error(f"🔴 ABNORMAL (Normal: {normal_prob*100:.2f}%, Abnormal: {abnormal_prob*100:.2f}%)")

# Overview table
st.markdown("**Overview probabilities**")
st.table({
    "Label": ["Normal", "Abnormal"],
    "Probability (%)": [
        f"{normal_prob*100:.2f}",
        f"{abnormal_prob*100:.2f}"
    ],
})

# Top classes
top = probs.argsort()[-3:][::-1]
st.markdown("**Top class probabilities**")
rows = [{"Class": CLASS_NAMES[i], "Probability (%)": f"{probs[i]*100:.2f}"} for i in top]
st.table(rows)

# -------------------------------------------------------------
# Grad-CAM
# -------------------------------------------------------------
def generate_gradcam(model, inp, cls):
    acts = {}
    grads = {}
    last = None

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m

    def forward(m, i, o): acts["v"] = o
    def backward(m, gi, go): grads["v"] = go[0]

    h1 = last.register_forward_hook(forward)
    h2 = last.register_backward_hook(backward)

    model.zero_grad()
    out = model(inp)
    out[0, cls].backward()

    h1.remove()
    h2.remove()

    A = acts["v"].detach().cpu().numpy()[0]
    G = grads["v"].detach().cpu().numpy()[0]

    weights = G.mean(axis=(1, 2))
    cam = np.zeros(A.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * A[i]

    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (224, 224))
    cam = np.uint8(cam * 255)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    return heatmap


st.markdown("---")
st.subheader("📊 Visual Insights")

left, right = st.columns(2)

with left:
    st.markdown("### 🌈 Grad-CAM")

    target = 0 if label == "Normal" else top[0]

    heatmap = generate_gradcam(clf_model, tensor, target)
    resized = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(resized, 0.6, heatmap, 0.4, 0)

    st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)

with right:
    st.markdown("### 🟡 YOLO Detection")
    result = det_model.predict(np.array(image))[0]
    st.image(result.plot(), use_column_width=True)

# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 👩‍💻 Developed by **Nandini💙** – Research Only")





































