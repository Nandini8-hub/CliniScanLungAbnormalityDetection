import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
# ------------------------------
# SAFE GLOBALS FOR PYTORCH 2.6
# ------------------------------
import torch
import torch.nn as nn
import torchvision
from torch.serialization import add_safe_globals

add_safe_globals([
    nn.Sequential,
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.Linear,
    torchvision.models.efficientnet.EfficientNet
])

# ------------------------------
# LOAD CLASSIFICATION MODEL
# ------------------------------
def load_classification_model():
    model = torch.load(
        "Script files/classification_model.pth",
        map_location="cpu",
        weights_only=False      # MUST BE FALSE
    )
    model.eval()
    return model


# =============================
# LOAD CLASSIFICATION MODEL
# =============================
class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.load("Script files/classification_model.pth", map_location="cpu")
        self.model.eval()

    def forward(self, x):
        return self.model(x)

# =============================
# LOAD DETECTION MODEL
# (YOLO / Torch Hub style)
# =============================
def load_detection_model():
    model = torch.load("Script files/detection_model.pt", map_location="cpu")
    return model


# =============================
# TRANSFORMS (224 × 224)
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_names = ["Normal", "Abnormal"]   # YOUR LABELS


# =============================
# GRAD-CAM IMPLEMENTATION
# =============================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradient = None
        self.activation = None

        def save_activation(module, inp, out):
            self.activation = out

        def save_gradient(module, grad_in, grad_out):
            self.gradient = grad_out[0]

        target_layer.register_forward_hook(save_activation)
        target_layer.register_backward_hook(save_gradient)

    def __call__(self, x):
        output = self.model(x)
        pred_class = output.argmax()
        self.model.zero_grad()
        output[0, pred_class].backward()

        gradient = self.gradient[0].detach().numpy()
        activation = self.activation[0].detach().numpy()

        weights = np.mean(gradient, axis=(1, 2))
        cam = np.zeros(activation.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activation[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))

        cam -= cam.min()
        cam /= cam.max()

        return cam, pred_class.item()


# =============================
# STREAMLIT UI
# =============================
st.title("🩺 CliniScan – Lung Abnormality Detection")
st.write("Upload a Chest X-Ray image for Classification + Detection + Grad-CAM")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(img).unsqueeze(0)

    # -----------------------------
    # CLASSIFICATION
    # -----------------------------
    st.subheader("🔍 Classification Result")

    clf_model = ClassificationModel()
    with torch.no_grad():
        output = clf_model(img_tensor)
        pred_class = output.argmax().item()
        st.success(f"Prediction: **{class_names[pred_class]}**")

    # -----------------------------
    # GRAD-CAM
    # -----------------------------
    st.subheader("🔥 Grad-CAM Explanation")

    # last conv layer of your model (usually model.features[-1] or model.layer4)
    # NOTE: Your .pth model already contains the architecture.
    # So we extract the last conv layer dynamically.
    last_conv = None
    for layer in reversed(list(clf_model.model.modules())):
        if isinstance(layer, nn.Conv2d):
            last_conv = layer
            break

    gradcam = GradCAM(clf_model.model, last_conv)
    cam, class_id = gradcam(img_tensor)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig = np.array(img.resize((224, 224)))
    superimposed = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    st.image(superimposed, caption="Grad-CAM Heatmap", use_column_width=True)

    # -----------------------------
    # DETECTION
    # -----------------------------
    st.subheader("📦 Detection Result (Bounding Boxes)")

    det_model = load_detection_model()

    # If YOLO-type model:
    try:
        results = det_model(img)
        results.render()
        st.image(results.ims[0], caption="Detection Output", use_column_width=True)
    except:
        st.warning("Detection model format not YOLO — showing raw prediction instead.")
        st.write(det_model(img_tensor))


st.write("---")
st.write("Made by **Nandini** 🩵")



   



# =============================
# LOAD CLASSIFICATION MODEL
# =============================
class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.load("Script files/classification_model.pth", map_location="cpu")
        self.model.eval()

    def forward(self, x):
        return self.model(x)

# =============================
# LOAD DETECTION MODEL
# (YOLO / Torch Hub style)
# =============================
def load_detection_model():
    model = torch.load("Script files/detection_model.pt", map_location="cpu")
    return model


# =============================
# TRANSFORMS (224 × 224)
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_names = ["Normal", "Abnormal"]   # YOUR LABELS


# =============================
# GRAD-CAM IMPLEMENTATION
# =============================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradient = None
        self.activation = None

        def save_activation(module, inp, out):
            self.activation = out

        def save_gradient(module, grad_in, grad_out):
            self.gradient = grad_out[0]

        target_layer.register_forward_hook(save_activation)
        target_layer.register_backward_hook(save_gradient)

    def __call__(self, x):
        output = self.model(x)
        pred_class = output.argmax()
        self.model.zero_grad()
        output[0, pred_class].backward()

        gradient = self.gradient[0].detach().numpy()
        activation = self.activation[0].detach().numpy()

        weights = np.mean(gradient, axis=(1, 2))
        cam = np.zeros(activation.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activation[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))

        cam -= cam.min()
        cam /= cam.max()

        return cam, pred_class.item()


# =============================
# STREAMLIT UI
# =============================
st.title("🩺 CliniScan – Lung Abnormality Detection")
st.write("Upload a Chest X-Ray image for Classification + Detection + Grad-CAM")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(img).unsqueeze(0)

    # -----------------------------
    # CLASSIFICATION
    # -----------------------------
    st.subheader("🔍 Classification Result")

    clf_model = ClassificationModel()
    with torch.no_grad():
        output = clf_model(img_tensor)
        pred_class = output.argmax().item()
        st.success(f"Prediction: **{class_names[pred_class]}**")

    # -----------------------------
    # GRAD-CAM
    # -----------------------------
    st.subheader("🔥 Grad-CAM Explanation")

    # last conv layer of your model (usually model.features[-1] or model.layer4)
    # NOTE: Your .pth model already contains the architecture.
    # So we extract the last conv layer dynamically.
    last_conv = None
    for layer in reversed(list(clf_model.model.modules())):
        if isinstance(layer, nn.Conv2d):
            last_conv = layer
            break

    gradcam = GradCAM(clf_model.model, last_conv)
    cam, class_id = gradcam(img_tensor)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig = np.array(img.resize((224, 224)))
    superimposed = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    st.image(superimposed, caption="Grad-CAM Heatmap", use_column_width=True)

    # -----------------------------
    # DETECTION
    # -----------------------------
    st.subheader("📦 Detection Result (Bounding Boxes)")

    det_model = load_detection_model()

    # If YOLO-type model:
    try:
        results = det_model(img)
        results.render()
        st.image(results.ims[0], caption="Detection Output", use_column_width=True)
    except:
        st.warning("Detection model format not YOLO — showing raw prediction instead.")
        st.write(det_model(img_tensor))


st.write("---")
st.write("Made by **Nandini** 🩵")



   


