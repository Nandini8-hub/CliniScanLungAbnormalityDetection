import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision
from torchvision import models
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="CliniScan - Lung Abnormality Detection", layout="wide")

st.title("🩺 CliniScan – Lung Abnormality Detection")
st.write("Upload a Chest X-Ray image for **Classification + Detection + Grad-CAM**")

# ---------------------------------------------------------
# MODEL PATHS (MATCHING YOUR GITHUB)
# ---------------------------------------------------------
CLASSIFICATION_MODEL_PATH = "Script files/classification_model.pth"
DETECTION_MODEL_PATH = "Script files/detection_model.pt"

# ---------------------------------------------------------
# SAFE GLOBALS FIX FOR EFFICIENTNET (IMPORTANT)
# ---------------------------------------------------------
torch.serialization.add_safe_globals([torchvision.models.efficientnet.EfficientNet])


# ---------------------------------------------------------
# LOAD CLASSIFICATION MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_classification_model():
    model = torch.load(CLASSIFICATION_MODEL_PATH, map_location="cpu")
    model.eval()
    return model


# ---------------------------------------------------------
# LOAD DETECTION MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_detection_model():
    model = YOLO(DETECTION_MODEL_PATH)
    return model


clf_model = load_classification_model()
det_model = load_detection_model()

# ---------------------------------------------------------
# TRANSFORMS (CHANGE SIZE IF YOU TRAINED DIFFERENT)
# ---------------------------------------------------------
img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# ---------------------------------------------------------
# CLASS LABELS (EDIT IF YOU HAVE MORE CLASSES)
# ---------------------------------------------------------
class_names = ["Normal", "Abnormal"]


# ---------------------------------------------------------
# GRAD-CAM IMPLEMENTATION
# ---------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        output_idx = output.argmax()

        self.model.zero_grad()
        output[0, output_idx].backward()

        grad = self.gradients[0].cpu().data.numpy()
        act = self.activations[0].cpu().data.numpy()

        weights = np.mean(grad, axis=(1, 2))
        cam = np.zeros(act.shape[1:], dtype=np.float32)

        for w, f in zip(weights, act):
            cam += w * f

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam, output_idx


# Initialize Grad-CAM for EfficientNet (last conv layer)
target_layer = clf_model.features[6][0]
gradcam = GradCAM(clf_model, target_layer)


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------------------------------------------------------
    # CLASSIFICATION
    # ---------------------------------------------------------
    st.subheader("🔍 Classification Result")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = clf_model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_label = class_names[predicted.item()]

    st.success(f"**Prediction:** {class_label}")

    # ---------------------------------------------------------
    # GRAD-CAM HEATMAP
    # ---------------------------------------------------------
    st.subheader("🔥 Grad-CAM Heatmap")

    cam, _ = gradcam.generate(input_tensor)

    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(np.array(image.resize((224, 224))), 0.6, heatmap, 0.4, 0)

    st.image(overlay, caption="Grad-CAM Visualization", use_container_width=True)

    # ---------------------------------------------------------
    # OBJECT DETECTION
    # ---------------------------------------------------------
    st.subheader("📦 Object Detection Result")

    results = det_model(image)

    annotated = results[0].plot()  # draw boxes

    st.image(annotated, caption="Detected Abnormalities", use_container_width=True)


