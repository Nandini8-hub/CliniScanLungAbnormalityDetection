# app.py - CliniScan final deploy-ready
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.models.efficientnet import EfficientNet
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import json
import os

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="CliniScan - Lung Abnormality Detection",
                   layout="wide", page_icon="🩺")
st.title("🩺 CliniScan - Lung Abnormality Detection Dashboard")

# -------------------------
# Paths (adjust if needed)
# -------------------------
CLASS_MODEL_PATH = os.path.join("Script files", "classification_model.pth")

# ▶️ FIX APPLIED HERE
DETECT_MODEL_PATH = os.path.join("Script files", "detection_model.pt")
# -------------------------

# -------------------------
# Helper: load class names from common files
# -------------------------
def load_class_names():
    candidates = [
        "classes.txt",
        "class_names.json",
        "chest_xray_coco_cleaned.json",
        "class_names.txt"
    ]
    for fn in candidates:
        if os.path.exists(fn):
            try:
                if fn.endswith(".json"):
                    j = json.load(open(fn, "r"))
                    if isinstance(j, dict) and "categories" in j:
                        return [c["name"] for c in j["categories"]]
                    if isinstance(j, dict) and "names" in j:
                        return j["names"]
                    if isinstance(j, list):
                        return j
                else:
                    with open(fn, "r") as f:
                        names = [line.strip() for line in f if line.strip()]
                        if names:
                            return names
            except Exception:
                continue

    fallback = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
        "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
        "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax", "Normal"
    ]
    return fallback

CLASS_NAMES = load_class_names()

# Sidebar control
st.sidebar.header("Settings")
st.sidebar.write("If the automatic class names are incorrect, edit them here or choose which label corresponds to 'Normal'.")

if st.sidebar.button("Show/Edit class names"):
    txt = st.sidebar.text_area("Edit class names (comma separated)", ", ".join(CLASS_NAMES), height=200)
    try:
        edited = [s.strip() for s in txt.split(",") if s.strip()]
        if edited:
            CLASS_NAMES = edited
            st.sidebar.success("Class names updated for this session.")
    except Exception:
        st.sidebar.error("Failed to parse class names.")

normal_default = CLASS_NAMES.index("Normal") if "Normal" in CLASS_NAMES else None
normal_label = st.sidebar.selectbox("Which class should be considered NORMAL?", options=CLASS_NAMES, index=(normal_default if normal_default is not None else 0))

# -------------------------
# Load classification model
# -------------------------
@st.cache_resource
def load_classification_model(path=CLASS_MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Classification model not found: {path}")
        return None
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

clf_model = load_classification_model()

# -------------------------
# Load detection model
# -------------------------
@st.cache_resource
def load_detection_model(path=DETECT_MODEL_PATH):
    if not os.path.exists(path):
        st.warning(f"Detection model not found at {path} — detection will be skipped.")
        return None
    try:
        y = YOLO(path)
        return y
    except Exception as e:
        st.error(f"Error loading detection model: {e}")
        return None

det_model = load_detection_model()

# -------------------------
# Grad-CAM
# -------------------------
def generate_gradcam(model, img_tensor, target_class):
    activations = {}
    gradients = {}

    last_conv = None
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
    score = out[0, target_class]
    score.backward()

    acts = activations.get("value")
    grads = gradients.get("value")

    try:
        fh.remove()
        bh.remove()
    except:
        pass

    if acts is None or grads is None:
        return None

    acts = acts.cpu().numpy()[0]
    grads = grads.cpu().numpy()[0]
    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    H = img_tensor.shape[2]; W = img_tensor.shape[3]
    cam = cv2.resize(cam, (W, H))
    cam = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

# -------------------------
# Upload image
# -------------------------
st.subheader("Upload a Chest X-Ray Image")
uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])

if uploaded_file is None:
    st.info("Upload an X-ray image to see classification, Grad-CAM and detection.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    # -------------------------
    # Classification
    # -------------------------
    if clf_model is None:
        st.warning("No classification model loaded.")
    else:
        st.subheader("Classification")
        with torch.no_grad():
            logits = clf_model(input_tensor)
            if isinstance(logits, torch.Tensor):
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            else:
                logits_tensor = logits[0] if isinstance(logits, (list,tuple)) else torch.tensor(logits)
                probs = F.softmax(logits_tensor, dim=1).cpu().numpy()[0]

        if len(CLASS_NAMES) != probs.shape[0]:
            st.warning(f"Class names count mismatch.")
            display_names = [f"Class_{i}" for i in range(probs.shape[0])]
        else:
            display_names = CLASS_NAMES

        idx = int(np.argmax(probs))
        label = display_names[idx]
        confidence = probs[idx]*100

        st.markdown(f"### 🔎 Predicted: **{label}** — Confidence: **{confidence:.2f}%**")

        if label == normal_label:
            st.success("✅ This X-ray is NORMAL")
        else:
            st.error("⚠️ This X-ray is ABNORMAL")

        st.write("Top 3 predictions:")
        top3 = probs.argsort()[-3:][::-1]
        for i in top3:
            name = display_names[i] if i < len(display_names) else f"Class_{i}"
            st.write(f"- {name}: {probs[i]*100:.2f}%")

        try:
            heatmap = generate_gradcam(clf_model, input_tensor, idx)
            if heatmap is not None:
                img_small = np.array(image.resize((input_tensor.shape[3], input_tensor.shape[2])))
                overlay = cv2.addWeighted(img_small, 0.6, heatmap, 0.4, 0)
                st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)
        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")

    # -------------------------
    # YOLO Detection
    # -------------------------
    if det_model is None:
        st.info("YOLO detection not available.")
    else:
        st.subheader("Detection (YOLOv8)")
        try:
            results = det_model.predict(np.array(image))
            annotated = results[0].plot()
            st.image(annotated, caption="YOLO Detection", use_column_width=True)

            boxes = []
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy()

                det_names = getattr(det_model.model, "names", None)
                if det_names is None:
                    det_names = getattr(det_model, "names", None)

                for b, c, cl in zip(xyxy, confs, cls_ids):
                    name = det_names[int(cl)] if (det_names and int(cl) < len(det_names)) else str(int(cl))
                    boxes.append({"box":[float(v) for v in b], "score": float(c), "class": name})

            if boxes:
                st.json(boxes)

        except Exception as e:
            st.error(f"YOLO detection failed: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("### 👩‍💻 Developed by **Nandini💙** — For research & educational use only.")


















