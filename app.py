import streamlit as st
import torch
from torchvision.models.efficientnet import EfficientNet
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from ultralytics import YOLO

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="CliniScan - Lung Abnormality Detection",
    layout="wide",
    page_icon="🩺"
)

st.title("🩺 CliniScan - Lung Abnormality Detection Dashboard")

# -------------------------
# Load Classification Model
# -------------------------
@st.cache_resource
def load_classification_model():
    model_path = r"Script files/classification_model.pth"  # adjust if needed
    torch.serialization.add_safe_globals([EfficientNet])
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        return None

clf_model = load_classification_model()

# -------------------------
# Load Detection Model
# -------------------------
@st.cache_resource
def load_detection_model():
    try:
        model = YOLO("models/detection_model.pt")  # adjust path if needed
        return model
    except Exception as e:
        st.error(f"Error loading detection model: {e}")
        return None

det_model = load_detection_model()

# -------------------------
# Grad-CAM Function
# -------------------------
def generate_gradcam(model, input_tensor, target_class=None):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Find last Conv2d layer
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        st.warning("No Conv2d layer found for Grad-CAM")
        return Image.new("RGB", (input_tensor.shape[3], input_tensor.shape[2]))

    last_conv.register_forward_hook(forward_hook)
    last_conv.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    loss = output[0, target_class]
    loss.backward()

    # Compute Grad-CAM
    gradient = gradients[0][0]
    activation = activations[0][0]
    weights = gradient.mean(dim=(1, 2))
    cam = torch.zeros(activation.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * activation[i]

    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear')
    cam = cam.squeeze().cpu().numpy()
    heatmap = np.uint8(255 * cam)
    heatmap_pil = Image.fromarray(heatmap).convert("L").resize((input_tensor.shape[3], input_tensor.shape[2]))
    return heatmap_pil

# -------------------------
# Image Upload
# -------------------------
st.subheader("Upload a Chest X-Ray Image")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------------------------
    # Classification Prediction
    # -------------------------
    if clf_model:
        st.subheader("Classification Prediction & Grad-CAM")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = clf_model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted_class = np.argmax(probs)

        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Class Probabilities: {probs}")

        # Grad-CAM visualization
        gradcam_heatmap = generate_gradcam(clf_model, input_tensor, target_class=predicted_class)
        gradcam_overlay = Image.blend(image.resize((224, 224)), gradcam_heatmap.convert("RGB"), alpha=0.5)
        st.image(gradcam_overlay, caption="Grad-CAM Overlay", use_column_width=True)

    # -------------------------
    # Detection Prediction
    # -------------------------
    if det_model:
        st.subheader("Detection Prediction")
        results = det_model.predict(np.array(image))
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="Detection Results", use_column_width=True)








