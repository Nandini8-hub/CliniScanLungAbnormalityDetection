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
DETECT_MODEL_PATH = os.path.join("models", "detection_model.pt")

# -------------------------
# Helper: load class names from common files
# -------------------------
def load_class_names():
    # check common files in repo
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
                    # try multiple shapes
                    if isinstance(j, dict) and "categories" in j:
                        return [c["name"] for c in j["categories"]]
                    if isinstance(j, dict) and "names" in j:
                        return j["names"]
                    if isinstance(j, list):
                        return j
                else:
                    # txt file: one per line
                    with open(fn, "r") as f:
                        names = [line.strip() for line in f if line.strip()]
                        if names:
                            return names
            except Exception:
                continue
    # fallback placeholder: 15 common chest x-ray labels (modify if needed)
    fallback = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
        "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
        "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax", "Normal"
    ]
    return fallback

CLASS_NAMES = load_class_names()

# Sidebar control: allow the user to override or pick which class means "Normal"
st.sidebar.header("Settings")
st.sidebar.write("If the automatic class names are incorrect, edit them here or choose which label corresponds to 'Normal'.")
# edit class names if needed
if st.sidebar.button("Show/Edit class names"):
    txt = st.sidebar.text_area("Edit class names (comma separated)", ", ".join(CLASS_NAMES), height=200)
    try:
        edited = [s.strip() for s in txt.split(",") if s.strip()]
        if edited:
            CLASS_NAMES = edited
            st.sidebar.success("Class names updated for this session.")
    except Exception:
        st.sidebar.error("Failed to parse class names.")

# choose normal label index (helps 'Normal/Abnormal' decision)
normal_default = CLASS_NAMES.index("Normal") if "Normal" in CLASS_NAMES else None
normal_label = st.sidebar.selectbox("Which class should be considered NORMAL?", options=CLASS_NAMES, index=(normal_default if normal_default is not None else 0))

# -------------------------
# Load classification model (safe for PyTorch 2.6+)
# -------------------------
@st.cache_resource
def load_classification_model(path=CLASS_MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Classification model not found: {path}")
        return None
    try:
        # Allowlist EfficientNet safe global for torch.load
        torch.serialization.add_safe_globals([EfficientNet])
    except Exception:
        # older torch may not have add_safe_globals
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
        # try to obtain names
        det_names = None
        try:
            # ultralytics has .model.names or .names
            det_names = getattr(y.model, "names", None)
            if det_names is None:
                det_names = getattr(y, "names", None)
        except Exception:
            det_names = None
        return y
    except Exception as e:
        st.error(f"Error loading detection model: {e}")
        return None

det_model = load_detection_model()

# -------------------------
# Grad-CAM (professional visualization)
# -------------------------
def generate_gradcam(model, img_tensor, target_class):
    """
    model: PyTorch model
    img_tensor: shape (1,3,H,W)
    target_class: int
    returns heatmap (H,W,3) as uint8 RGB
    """
    activations = {}
    gradients = {}

    # find last conv layer
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

    # forward & backward
    model.zero_grad()
    out = model(img_tensor)  # logits
    if out.ndim == 1:
        out = out.unsqueeze(0)
    score = out[0, target_class]
    score.backward()

    # fetch
    acts = activations.get("value")  # tensor shape (B, C, H, W)
    grads = gradients.get("value")
    # remove hooks
    try:
        fh.remove()
        bh.remove()
    except Exception:
        pass

    if acts is None or grads is None:
        return None

    acts = acts.cpu().numpy()[0]   # (C,H,W)
    grads = grads.cpu().numpy()[0] # (C,H,W)
    weights = np.mean(grads, axis=(1,2))  # (C,)
    cam = np.zeros(acts.shape[1:], dtype=np.float32)  # (H,W)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    H = img_tensor.shape[2]; W = img_tensor.shape[3]
    cam = cv2.resize(cam, (W, H))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

# -------------------------
# UI: Upload image
# -------------------------
st.subheader("Upload a Chest X-Ray Image")
uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])

if uploaded_file is None:
    st.info("Upload an X-ray image to see classification, Grad-CAM and detection.")
else:
    # load & show
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocess for classifier
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)  # (1,3,224,224)

    # -------------------------
    # Classification
    # -------------------------
    if clf_model is None:
        st.warning("No classification model loaded. Please add classification_model.pth at Script files/.")
    else:
        st.subheader("Classification")
        with torch.no_grad():
            logits = clf_model(input_tensor)
            if isinstance(logits, torch.Tensor):
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            else:
                # some models return tuple
                logits_tensor = logits[0] if isinstance(logits, (list,tuple)) else torch.tensor(logits)
                probs = F.softmax(logits_tensor, dim=1).cpu().numpy()[0]

        # Ensure class names length matches output
        if len(CLASS_NAMES) != probs.shape[0]:
            st.warning(f"The number of class names ({len(CLASS_NAMES)}) does not match model output ({probs.shape[0]}). Please update class names file or edit them in the sidebar.")
            # if lengths differ, create generic numeric names
            display_names = [f"Class_{i}" for i in range(probs.shape[0])]
        else:
            display_names = CLASS_NAMES

        idx = int(np.argmax(probs))
        label = display_names[idx]
        confidence = probs[idx]*100.0

        st.markdown(f"### 🔎 Predicted: **{label}**  —  Confidence: **{confidence:.2f}%**")

        # Normal / Abnormal decision using sidebar selection
        is_normal = (label == normal_label)
        if is_normal:
            st.success("✅ This X-ray is NORMAL")
        else:
            st.error("⚠️ This X-ray is ABNORMAL")

        # Top-3
        st.write("Top 3 predictions:")
        top3 = probs.argsort()[-3:][::-1]
        for i in top3:
            name = display_names[i] if i < len(display_names) else f"Class_{i}"
            st.write(f"- {name}: {probs[i]*100:.2f}%")

        # Grad-CAM (produce heatmap + overlay)
        try:
            heatmap = generate_gradcam(clf_model, input_tensor, idx)
            if heatmap is not None:
                img_small = np.array(image.resize((input_tensor.shape[3], input_tensor.shape[2])))
                overlay = cv2.addWeighted(img_small, 0.6, heatmap, 0.4, 0)
                st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)
            else:
                st.info("Grad-CAM not available for this model.")
        except Exception as e:
            st.error(f"Grad-CAM generation failed: {e}")

    # -------------------------
    # Detection (YOLO)
    # -------------------------
    if det_model is None:
        st.info("YOLO detection not available (no detection model).")
    else:
        st.subheader("Detection (YOLOv8)")
        try:
            results = det_model.predict(np.array(image))
            annotated = results[0].plot()
            # try to get class names from model
            det_names = getattr(det_model.model, "names", None)
            if det_names is None:
                det_names = getattr(det_model, "names", None)
            if det_names:
                st.write("Detection class names loaded from YOLO model.")
            st.image(annotated, caption="YOLOv8 Detection", use_column_width=True)
            # also show raw detections table
            boxes = []
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy()
                for b, c, cl in zip(xyxy, confs, cls_ids):
                    name = det_names[int(cl)] if (det_names and int(cl) < len(det_names)) else str(int(cl))
                    boxes.append({"box":[float(v) for v in b], "score": float(c), "class": name})
            if boxes:
                st.write("Detections:")
                st.json(boxes)
        except Exception as e:
            st.error(f"YOLO detection failed: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("### 👩‍💻 Developed by **Nandini** — For research & educational use only.")















