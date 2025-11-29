# ------------------------------
# SAFE GLOBALS FOR PYTORCH 2.6
# ------------------------------
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
# LOAD CLASSIFICATION MODEL
# (EfficientNet + state_dict)
# =============================
def load_classification_model():
    # 1. Create architecture
    model = torchvision.models.efficientnet_b0(pretrained=False)

    # 2. Adjust classifier for your labels
    model.classifier[1] = nn.Linear(1280, 2)   # Normal / Abnormal

    # 3. Load ONLY the weights (state_dict)
    state_dict = torch.load("Script files/classification_model.pth", map_location="cpu")

    model.load_state_dict(state_dict)
    model.eval()
    return model


# =============================
# CLASSIFICATION MODEL WRAPPER
# =============================
class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Build EfficientNet architecture
        self.model = torchvision.models.efficientnet_b0(pretrained=False)
        self.model.classifier[1] = nn.Linear(1280, 2)

        # Load the state_dict
        state_dict = torch.load("Script files/classification_model.pth", map_location="cpu")
        self.model.load_state_dict(state_dict)

        self.model.eval()

    def forward(self, x):
        return self.model(x)


# =============================
# LOAD DETECTION MODEL
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

class_names = ["Normal", "Abnormal"]


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

        gradient = self.gradient[0].detach(




   







