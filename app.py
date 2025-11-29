# -------------------------
# CliniScan – Lung Abnormality Detection
# Full app.py — Ready to run
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
# Page config
# -------------------------
st.set_page_config(
    page_title="CliniScan - Lung Abnormality Detection",
    layout="wide",
    page_icon="🩺"
)

# -------------------------
# Simple CSS for visuals
# -------------------------
st.markdown(
    """
    <style>
    .logo { width:140px; display:block; margin:auto; border-radius:10px; }
    .header-banner { padding:12px; background:#eaf3ff; border-radius:10px; color:#003f8c; text-align:center; font-weight:600; }
    .condition-box { background:#fff; border-radius:12px; padding:18px; box-shadow:0 0 8px rgba(0,0,0,0.06); }
    table td, table th { padding:6px 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Paths (adjust as needed)
# -------------------------
CLASS_MODEL_PATH = os.path.join("Script files", "classification_model.pth")
DETECT_MODEL_PATH = os.path.join("Script files", "detection_model.pt")

# -------------------------
# Class names (must match your classifier)
# Normal is assumed at index 0
# -------------------------
CLASS_NAMES = [
    "Normal", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# -------------------------
# Sidebar: logo, settings
# -------------------------
with st.sidebar:
    # try to use a local logo if present, otherwise use a web fallback
    if os.path.exists("streamlit_logo.png"):
        st.image("streamlit_logo.png", use_column_width=False, output_format="PNG", caption="")
    else:
        # fallback image (external)
        st.image("https://i.imgur.com/E66QGZo.png", use_column_width=False, caption="")

    st.title("🩺 CliniScan")
    st.markdown(
        """
        **Features**
        - Fast Chest X-ray analysis
        - Grad-CAM visualisations
        - YOLO detection (optional)

        **Use**: Research / educational only
        """
    )

    st.markdown("---")
    st.markdown("### Prediction settings")
    normal_default = CLASS_NAMES.index("Normal")
    # threshold to decide Normal vs Abnormal (you can tune)
    normal_threshold = st.slider("Normal threshold (normal_prob >= ?)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

# -------------------------
# Load models (cached)
# -------------------------
@st.cache_resource
def load_classification_model(path=CLASS_MODEL_PATH):
    if not os.path.exists(path):
        return None
    # allow efficientnet class to be safe for torch.load if needed
    try:
        torch.serialization.add_safe_globals([EfficientNet])
    except Exception:
        pass
    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        return None

@st.cache_resource
def load_detection_model(path=DETECT_MODEL_PATH):
    if not os.path.exists(path):
        return None
    try:
        return YOLO(path)
    except Exception as e:
        st.warning(f"YOLO load error: {e}")
        return None

clf_model = load_classification_model()
det_model = load_detection_model()

# -------------------------
# Header / homepage
# -------------------------
st.markdown("<div class='header-banner'>Model Loaded | Accuracy: <b>95.37%</b></div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center; margin-top:10px;'>🩺 CliniScan — Lung Abnormality Detection</h1>", unsafe_allow_html=True)

st.markdown("<div class='condition-box'>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>✨ Detectable Conditions</h3>", unsafe_allow_html=True)

condition_list = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "Interstitial Lung Disease (ILD)", "Infiltration",
    "Lung Opacity", "Nodule / Mass", "Other lesion", "Pleural effusion",
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis", "Edema"
]
st.markdown("<ul>", unsafe_allow_html=True)
for c in condition_list:
    st.markdown(f"<li style='font-size:16px'>{c}</li>", unsafe_allow_html=True)
st.markdown("</ul></div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# Upload image
# -------------------------
st.subheader("📤 Upload a Chest X-ray Image")
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
if uploaded_file is None:
    st.info("Upload an X-ray image to see classification, Grad-CAM and detection.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", use_column_width=True)

# -------------------------
# Preprocess
# -------------------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
input_tensor = transform(image).unsqueeze(0)  # shape [1, C, H, W]

# -------------------------
# Classification inference with Normal vs Abnormal handling
# -------------------------
st.subheader("🔍 Classification Results")

if clf_model is None:
    st.error("Classification model not found. Please place classification_model.pth under 'Script files'.")
else:
    with torch.no_grad():
        logits = clf_model(input_tensor)
        # handle various possible output shapes
        if isinstance(logits, tuple) or isinstance(logits, list):
            logits = logits[0]
        probs_tensor = torch.softmax(logits, dim=1)[0]  # 1D tensor for classes
        probs = probs_tensor.cpu().numpy()

    # Normal probability (class index 0)
    normal_prob = float(probs[0])
    # Abnormal = sum of all other class probabilities
    abnormal_prob = float(probs[1:].sum())

    # Decide label using threshold and which is larger
    is_normal = (normal_prob >= normal_threshold) and (normal_prob >= abnormal_prob)
    final_label = "Normal" if is_normal else "Abnormal"

    # Display prediction
    if final_label == "Normal":
        st.success(f"🟢 Prediction: **{final_label}** ({normal_prob*100:.2f}%)")
    else:
        st.error(f"🔴 Prediction: **{final_label}** (Normal: {normal_prob*100:.2f}%, Abnormal: {abnormal_prob*100:.2f}%)")

    # Show a small table: Normal vs Abnormal probs
    st.markdown("**Overview probabilities**")
    prob_table = {
        "Label": ["Normal", "Abnormal (sum of others)"],
        "Probability (%)": [f"{normal_prob*100:.2f}", f"{abnormal_prob*100:.2f}"],
    }
    st.table(prob_table)

    # Top-3 class probabilities
    topk = 3
    topk_idx = probs.argsort()[-topk:][::-1]
    st.markdown("**Top class probabilities**")
    rows = []
    for i in topk_idx:
        rows.append({"Class": CLASS_NAMES[i], "Probability (%)": f"{probs[i]*100:.2f}"})
    st.table(rows)

# -------------------------
# Grad-CAM implementation
# -------------------------
def generate_gradcam(model, img_tensor, target_class):
    """
    Grad-CAM: finds last conv layer, attach hooks, compute cam.
    Returns heatmap in RGB (H,W,3) in same size as input_tensor spatial dims.
    """
    activations = {}
    gradients = {}
    last_conv = None

    # find last Conv2d layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    if last_conv is None:
        return None

    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_backward_hook(backward_hook)

    model.zero_grad()
    out = model(img_tensor)
    if out.ndim == 1:
        out = out.unsqueeze(0)
    score = out[0, int(target_class)]
    score.backward()

    acts = activations.get("value")
    grads = gradients.get("value")

    try:
        fh.remove()
        bh.remove()
    except Exception:
        pass

    if acts is None or grads is None:
        return None

    acts = acts.cpu().numpy()[0]
    grads = grads.cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    cam = cv2.resize(cam, (W, H))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

# -------------------------
# Display Grad-CAM and YOLO side-by-side
# -------------------------
st.markdown("---")
st.subheader("📊 Visual Insights")

col1, col2 = st.columns([1, 1])

# Left: Grad-CAM
with col1:
    st.markdown("### 🌈 Grad-CAM")
    try:
        # pick target class: if normal show class 0 else show top abnormal class
        if clf_model is None:
            st.info("No classification model — can't generate Grad-CAM.")
        else:
            if final_label == "Normal":
                target_cls = 0
            else:
                # pick the highest-probability abnormal class
                if probs.shape[0] > 1:
                    target_cls = int(probs[1:].argmax() + 1)
                else:
                    target_cls = 0

            heatmap = generate_gradcam(clf_model, input_tensor, target_cls)
            if heatmap is None:
                st.info("Grad-CAM not available for this model architecture.")
            else:
                img_small = np.array(image.resize((input_tensor.shape[3], input_tensor.shape[2])))
                overlay = cv2.addWeighted(img_small, 0.6, heatmap, 0.4, 0)
                st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")

# Right: YOLO detection
with col2:
    st.markdown("### 🟡 YOLO Detection")
    if det_model is None:
        st.info("YOLO detection model not found — detection skipped.")
    else:
        try:
            results = det_model.predict(np.array(image))
            annotated = results[0].plot()
            st.image(annotated, caption="YOLO Detection", use_column_width=True)

            # build boxes list (x1,y1,x2,y2, score, class)
            boxes = []
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy()
                det_names = getattr(det_model.model, "names", None) or getattr(det_model, "names", None)
                for b, c, cl in zip(xyxy, confs, cls_ids):
                    name = det_names[int(cl)] if (det_names and int(cl) < len(det_names)) else str(int(cl))
                    boxes.append({"class": name, "confidence": float(c) * 100, "box": [float(x) for x in b]})
            if boxes:
                st.json(boxes)
            else:
                st.info("No detections found.")
        except Exception as e:
            st.error(f"YOLO detection failed: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("### 👩‍💻 Developed by **Nandini 💙** — For educational & research use only.")


































