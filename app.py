# ================================
# 1. IMPORTS
# ================================
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.serialization import add_safe_globals

# Allow Sequential for safe unpickling (PyTorch 2.6 fix)
add_safe_globals([nn.Sequential])

# ================================
# 2. CLASS LABELS (EDIT IF NEEDED)
# ================================
class_names = ["Normal", "Abnormal"]   # <-- your dataset

# ================================
# 3. IMAGE TRANSFORMS
# (USE SAME SIZE AS TRAINING)
# ================================
img_size = 224   # <-- you trained on 224x224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================================
# 4. LOAD MODEL ARCHITECTURE
# ================================
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

# ================================
# 5. LOAD TRAINED WEIGHTS
# ================================
checkpoint_path = "model.pth"   # <-- your repo file

ckpt = torch.load(checkpoint_path, weights_only=True)
model.load_state_dict(ckpt)
model.eval()

print("Model loaded successfully!")


# ================================
# 6. PREDICT FUNCTION
# ================================
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    inp = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(inp)
        _, predicted = torch.max(outputs, 1)

    pred_class = class_names[predicted.item()]
    print(f"\nPredicted Class: {pred_class}")

    return img, inp, predicted.item()


# ================================
# 7. GRAD-CAM IMPLEMENTATION
# ================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.target_layer = target_layer

        # Hook for gradients
        target_layer.register_backward_hook(self.save_gradients)

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x):
        # Forward pass
        features = None

        def forward_hook(module, input, output):
            nonlocal features
            features = output

        hook = self.target_layer.register_forward_hook(forward_hook)
        output = self.model(x)
        hook.remove()

        pred_idx = output.argmax().item()

        # Backward pass
        self.model.zero_grad()
        output[0, pred_idx].backward()

        # Grad-CAM calculation
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        cam = torch.zeros(features.shape[2:], dtype=torch.float32)

        for i, w in enumerate(pooled_gradients):
            cam += w * features[0, i, :, :]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam.detach().numpy()

        return cam, pred_idx


# Select LAST CNN layer of ResNet18
target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)


# ================================
# 8. PREDICT + GRAD-CAM VISUALIZATION
# ================================
def run_gradcam(image_path):
    img, inp, pred = predict_image(image_path)

    cam, _ = gradcam(inp)

    img_np = np.array(img)

    cam_resized = np.uint8(255 * cam)
    cam_resized = np.stack([cam_resized] * 3, axis=-1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(img_np)
    plt.imshow(cam_resized, cmap='jet', alpha=0.45)
    plt.axis("off")

    plt.show()


# ================================
# 9. RUN EVERYTHING (EDIT FILE PATH)
# ================================
image_path = "output/single_image.jpg"   # <-- choose your image
run_gradcam(image_path)

