import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="CliniScan - Lung Abnormality Detection",
    layout="wide",
    page_icon="🩺"
)

st.title("🩺 CliniScan - Lung Abnormality Detection System")

# -------------------------
# Load Models (UPDATED: Removed 'models/' folder name)
# -------------------------
@st.cache_resource
def load_detection_model():
    return YOLO("Script files/detection_model.pt")  # UPDATED

@st.cache_resource
def load_classification_model():
    model = resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 2)
    model.load_state_dict(torch.load("script files/classification_model.pth", map_location="cpu"))  # UPDATED
    model.eval()
    return model


detection_model = load_detection_model()
classification_model = load_classification_model()

# -------------------------
# Preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------------
# GradCAM
# -------------------------
def generate_gradcam(model, img_tensor, target_layer="layer4"):
    model.eval()
    extractor = create_feature_extractor(model, return_nodes={target_layer: "feat"})
    features = extractor(img_tensor.unsqueeze(0))["feat"]

    heatmap = torch.mean(features[0], dim=0).detach().numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()
    return heatmap

# -------------------------
# Upload Section
# -------------------------
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(img)

    # -------------------------
    # YOLO Detection
    # -------------------------
    st.subheader("🔍 Detection Results")
    results = detection_model(img_np)
    res_img = results[0].plot()
    st.image(res_img, caption="Detected Abnormalities", use_column_width=True)

    # -------------------------
    # Classification
    # -------------------------
    st.subheader("🧬 Classification Results")

    input_tensor = transform(img)
    output = classification_model(input_tensor.unsqueeze(0))
    prob = torch.softmax(output, dim=1)
    pred_class = torch.argmax(prob).item()

    classes = ["Normal", "Abnormal"]
    st.write(f"### **Prediction: {classes[pred_class]}**")
    st.write(f"Confidence: {prob[0][pred_class].item():.4f}")

    # -------------------------
    # Grad-CAM Visualization
    # -------------------------
    st.subheader("🔥 Grad-CAM Heatmap")

    heatmap = generate_gradcam(classification_model, input_tensor)

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(heatmap, 0.5, cv2.resize(img_np, (224,224)), 0.5, 0)

    st.image(blended, caption="Grad-CAM Heatmap", use_column_width=False)


