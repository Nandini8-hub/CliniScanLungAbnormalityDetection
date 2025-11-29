# -------------------------  
#  CliniScan – Lung Abnormality Detection  
#  FULL app.py (Deploy Ready)  
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
st.set_page_config(page_title="CliniScan - Lung Abnormality Detection",
                   layout="wide", page_icon="🩺")
st.title("🩺 CliniScan - Lung Abnormality Detection Dashboard")

# -------------------------
# Paths
# -------------------------
CLASS_MODEL_PATH = os.path.join("Script files", "classification_model.pth")
DETECT_MODEL_PATH = os.path.join("Script files", "detection_model.pt")

# -------------------------
# Class Names (15-classes)
# -------------------------
CLASS_NAMES = [
    "Normal", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# Sidebar select normal class
st.sidebar.header("Settings")
normal_default = CLASS_NAMES.index("Normal")
normal_label = st.sidebar.selectbox(
    "Which class should be considered NORMAL?",
    CLASS_NAMES,
    index=normal_default
)

# -------------------------
# Load Classification Model
# -------------------------
@st.cache_resource
def load_classification_model(path=CLASS_MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Classification model not found: {path}")
        return None

    try:
        torch.serialization.add_safe_globals([EfficientNet])
    except:
        pass

    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        return None

clf_model = load_classification_model()

# -------------------------
# Load YOLO Detection Model
# -------------------------
@st.cache_resource
def load_detection_model(path=DETECT_MODEL_PATH):
    if not os.path.exists(path):
        st.warning(f"Detection model not found at {path} — skipping YOLO.")
        return None
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Error loading detection model: {e}")
        return None

det_model = load_detection_model()

# -------------------------
# Grad-CAM
# -------------------------
def generate_gradcam(model, img_tensor, target_class):
    activations, gradients = {}, {}
    last_conv = None

    # find last conv layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    if last_conv is None:
        return None

    # hooks
    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_backward_hook(backward_hook)

    # forward
    model.zero_grad()
    out = model(img_tensor)
    if out.ndim == 1:
        out = out.unsqueeze(0)
    score = out[0, target_class]
    score.backward()

    acts = activations.get("value")
    grads = gradients.get("value")

    fh.remove()
    bh.remove()

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
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# -------------------------
# Upload Image
# -------------------------
st.subheader("Upload a Chest X-Ray Image")
uploaded_file = st.file_uploader("Choose X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload an image to continue.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", use_column_width=True)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)

# -------------------------
# Classification
# -------------------------
st.subheader("Classification Results")

if clf_model is None:
    st.error("Classification model missing.")
else:
    with torch.no_grad():
        logits = clf_model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    pred_conf = probs[pred_idx] * 100

    st.markdown(f"### 🔍 Predicted: **{pred_label}** — {pred_conf:.2f}%")

    # Normal / Abnormal logic
    if pred_label == normal_label and pred_conf > 50:
        st.success("✅ NORMAL")
    else:
        st.error("⚠️ ABNORMAL")

    # Top 3 predictions
    st.write("Top 3 predictions:")
    top3 = probs.argsort()[-3:][::-1]
    for t in top3:
        st.write(f"- **{CLASS_NAMES[t]}:** {probs[t]*100:.2f}%")

# -------------------------
# SIDE-BY-SIDE: GRAD-CAM & DETECTION
# -------------------------
st.markdown("---")
st.subheader("📊 Visual Insights")

col1, col2 = st.columns([1, 1])

# -------- LEFT : GRAD-CAM --------
with col1:
    st.markdown("### 🌈 Grad-CAM Insights")
    try:
        heatmap = generate_gradcam(clf_model, input_tensor, pred_idx)
        if heatmap is not None:
            img_small = np.array(image.resize((input_tensor.shape[3], input_tensor.shape[2])))
            overlay = cv2.addWeighted(img_small, 0.6, heatmap, 0.4, 0)
            st.image(overlay, caption="Grad-CAM Visualization", use_column_width=True)
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")

# -------- RIGHT : YOLO DETECTION --------
with col2:
    st.markdown("### 🟡 Detected Regions")

    if det_model is None:
        st.warning("YOLO model not available.")
    else:
        results = det_model.predict(np.array(image))
        annotated = results[0].plot()
        st.image(annotated, caption="Detection Bounding Boxes", use_column_width=True)

        result = results[0]
        detections = []

        if hasattr(result, "boxes") and result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()
            names = det_model.model.names

            for b, c, cl in zip(xyxy, confs, cls_ids):
                detections.append({
                    "Condition": names[int(cl)],
                    "Confidence": round(float(c)*100, 2),
                    "Status": "🟢" if c > 0.5 else "🔴"
                })

        st.markdown("### 📝 Top Findings")
        if detections:
            st.dataframe(detections, use_container_width=True)
        else:
            st.info("No findings detected.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("### 👩‍💻 Developed by **Nandini 💙** — For educational & research use only.")






















