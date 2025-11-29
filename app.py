
import streamlit as st
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
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

# =============================
# CLASSIFICATION MODEL
# =============================
class ClassificationModel(nn.Module):
    def _init_(self):
        super()._init_()

        self.model = torchvision.models.efficientnet_b0(pretrained=False)
        self.model.classifier[1] = nn.Linear(1280, 2)

        state_dict = torch.load("Script files/classification_model.pth", map_location="cpu")
        self.model.load_state_dict(state_dict)

        self.model.eval()

    def forward(self, x):
        return self.model(x)


# =============================
# DETECTION MODEL
# =============================
def load_detection_model():
    return torch.load("Script files/detection_model.pt", map_location="cpu")


# =============================
# TRANSFORMS
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_names = ["Normal", "Abnormal"]


# =============================
# GRAD-CAM
# =============================
class GradCAM:
    def _init_(self, model, target_layer):
        self.model = model
        self.gradient = None
        self.activation = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activation = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradient = grad_out[0]

    def _call_(self, x):
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

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(img).unsqueeze(0)

    # ---- CLASSIFICATION ----
    clf = ClassificationModel()
    with torch.no_grad():
        output = clf(img_tensor)
        pred = output.argmax().item()
        st.success(f"Prediction: {class_names[pred]}")

    # ---- GRAD-CAM ----
    st.subheader("🔥 Grad-CAM")
    last_conv = next(m for m in reversed(list(clf.model.modules())) if isinstance(m, nn.Conv2d))
    cam_gen = GradCAM(clf.model, last_conv)

    cam, _ = cam_gen(img_tensor)
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    orig = np.array(img.resize((224, 224)))
    final = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
    st.image(final, caption="Grad-CAM Heatmap")

    # ---- DETECTION ----
    st.subheader("📦 Detection")
    det = load_detection_model()

    try:
        results = det(img)
        results.render()
        st.image(results.ims[0], caption="Detection Output")
    except:
        st.warning("Detection model not YOLO")
        st.write(det(img_tensor))

st.write("---")
st.write("Made by Nandini 🩵")
