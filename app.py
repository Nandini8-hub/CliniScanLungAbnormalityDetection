import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2

# ============================
#        GRAD-CAM CLASS
# ============================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor):
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1)

        self.model.zero_grad()
        output[0, pred_class].backward()

        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
        activations = self.activations

        cam = (gradients * activations).sum(dim=1).squeeze().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam, pred_class.item()


# ============================
#        LOAD MODELS
# ============================
@st.cache_resource
def load_classification_model():
    model = torch.load("classification_model.pth", map_location="cpu")
    model.eval()
    return model

@st.cache_resource
def load_detection_model():
    from ultralytics import YOLO
    return YOLO("detection_model.pt")


# ============================
#      PREPROCESSING
# ============================
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)


# ============================
#          STREAMLIT UI
# ============================
st.title("🫁 CliniScan – Lung Abnormality Detection & Grad-CAM")

option = st.selectbox(
    "Choose Mode:",
    ["Classification", "Object Detection", "Grad-CAM (Heatmap)"]
)

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    # ===============================================
    #                CLASSIFICATION
    # ===============================================
    if option == "Classification":
        st.subheader("🔍 Classification Result")

        class_names = ["Normal", "Abnormal"]

        model = load_classification_model()
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)

        st.success(f"### 🩺 Prediction: **{class_names[pred.item()]}**")

    # ===============================================
    #                OBJECT DETECTION
    # ===============================================
    elif option == "Object Detection":
        st.subheader("🩻 Object Detection Result")

        det_model = load_detection_model()
        results = det_model(image)

        result_img = results[0].plot()
        st.image(result_img, caption="Detection Output", width=600)

    # ===============================================
    #                GRAD-CAM (LAST)
    # ===============================================
    elif option == "Grad-CAM (Heatmap)":
        st.subheader("🔥 Grad-CAM Heatmap (Last Step)")

        model = load_classification_model()
        input_tensor = preprocess_image(image)

        # auto-detect last Conv layer
        target_layer = None
        for layer in reversed(list(model.modules())):
            if isinstance(layer, torch.nn.Conv2d):
                target_layer = layer
                break

        cam_gen = GradCAM(model, target_layer)
        cam, pred = cam_gen.generate_cam(input_tensor)

        cam = cv2.resize(cam, (image.width, image.height))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

        st.image(overlay, caption="Grad-CAM Heatmap", width=500)

