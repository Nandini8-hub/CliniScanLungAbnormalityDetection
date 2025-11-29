import streamlit as st
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision
from torch.serialization import add_safe_globals

# Safe globals for PyTorch 2.6
add_safe_globals([
    nn.Sequential,
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.Linear,
    torchvision.models.efficientnet.EfficientNet
])

# =============================
# TRANSFORMS
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_names = ["Normal", "Abnormal"]

# =============================
# BUILD & LOAD CLASSIFICATION MODEL
# =============================
def load_classification_model():
    # Build EfficientNet architecture
    model = torchvision.models.efficientnet_b0(pretrained=False)

    # Adjust classifier (2 classes)
    model.classifier[1] = nn.Linear(1280, 2)

    # Load state_dict
    state_dict = torch.load(r"Script files/classification_model.pth", map_location="cpu")

    model.load_state_dict(state_dict)

    model.eval()
    return model

# =============================
# GRAD-CAM IMPLEMENTATION
# =============================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradient = None
        self.activation = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activation = out

    def save_gradient(self, module, grad_in, grad_out):
        self.gradient = grad_out[0]

    def __call__(self, x):
        output = self.model(x)
        pred_class = output.argmax()

        self.model.zero_grad()
        output[0, pred_class].backward()

        grad = self.gradient[0].detach().numpy()
        act = self.activation[0].detach().numpy()

        weights = np.mean(grad, axis=(1, 2))
        cam = np.zeros(act.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * act[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        cam /= cam.max()

        return cam, pred_class.item()

# =============================
# STREAMLIT APP
# =============================
st.title("🩺 CliniScan – Lung Abnormality Detection")
st.write("Upload a Chest X-Ray image for Classification + Detection + Grad-CAM")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(img).unsqueeze(0)

    # ---- CLASSIFICATION ----
    st.subheader("🔍 Classification Result")

    clf = load_classification_model()

    with torch.no_grad():
        output = clf(img_tensor)
        pred = output.argmax().item()
        st.success(f"Prediction: *{class_names[pred]}*")

    # ---- GRAD-CAM ----
    st.subheader("🔥 Grad-CAM Explanation")

    last_conv = next(m for m in reversed(list(clf.modules())) if isinstance(m, nn.Conv2d))
    cam_gen = GradCAM(clf, last_conv)
    cam, _ = cam_gen(img_tensor)

    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    orig = np.array(img.resize((224, 224)))
    superimposed = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
    st.image(superimposed, caption="Grad-CAM Heatmap", use_column_width=True)

    # ---- DETECTION ----
    st.subheader("📦 Detection Result")

    det_model = torch.load("Script files/detection_model.pt", map_location="cpu")

    try:
        results = det_model(img)
        results.render()
        st.image(results.ims[0], caption="Detection Output", use_column_width=True)
    except:
        st.warning("Detection model not YOLO — showing raw output")
        st.write(det_model(img_tensor))

st.write("---")
st.write("Made by Nandini 🩵")





