

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import cv2


def preprocess_image_pil(pil_image, image_size=(224, 224)):
    """Return input_tensor (1,C,H,W) and normalized RGB float image (H,W,3) in [0,1]."""
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_t = transform(pil_image).unsqueeze(0)  # 1,C,H,W
    # Also create an RGB image in [0,1] that can be used for overlay
    rgb = np.array(pil_image.resize(image_size)).astype(np.float32) / 255.0
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    return img_t, rgb


def get_target_layer(model):
    """
    Try to pick a good target layer for Grad-CAM. Heuristic: return the last convolutional layer.
    If it fails, user can pass target_layer explicitly.
    """
    # Search modules in reverse order
    for name, module in reversed(list(model.named_modules())):
        classname = module.__class__.__name__.lower()
        if 'conv' in classname:
            return module
    # fallback to last module
    mods = list(model.modules())
    if len(mods) > 0:
        return mods[-1]
    raise RuntimeError("Could not find a target layer in the model. Pass target_layer explicitly.")


def _apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    activation_map: HxW float in [0,1]
    org_img: HxWx3 float in [0,1]
    return: overlayed rgb uint8
    """
    heatmap = (activation_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, colormap)  # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0
    overlay = heatmap * alpha + org_img * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


def generate_gradcam(model, input_tensor, target_layer=None, target_category=None, use_cuda=False):
    """
    Generate Grad-CAM heatmap for a single input_tensor: 1xCxHxW.
    Returns: grayscale_cam (HxW float in [0,1]) , overlay_rgb (HxWx3 uint8)
    Tries to use pytorch_grad_cam if available for robustness.
    """
    # Ensure eval mode
    model.eval()
    device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    try:
        # Use pytorch_grad_cam if installed
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        if target_layer is None:
            target_layer = get_target_layer(model)
        cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type == 'cuda'))
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        # reconstruct normalized rgb image
        inp = input_tensor.detach().cpu()[0]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb_img = inp * std + mean
        rgb_img = np.transpose(rgb_img.numpy(), (1, 2, 0))
        rgb_img = np.clip(rgb_img, 0, 1)
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return grayscale_cam, overlay
    except Exception:
        # Fallback lightweight CAM implementation using hooks
        activations = {}
        gradients = {}
        if target_layer is None:
            target_layer = get_target_layer(model)

        def forward_hook(module, inp, out):
            activations['value'] = out.detach()

        def backward_hook(module, grad_in, grad_out):
            gradients['value'] = grad_out[0].detach()

        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_backward_hook(backward_hook)

        output = model(input_tensor)
        if isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        if target_category is None:
            target_category = int(logits.argmax(dim=1).item())

        loss = logits[0, target_category]
        model.zero_grad()
        loss.backward(retain_graph=True)

        pooled_grads = torch.mean(gradients['value'], dim=[0, 2, 3])  # C
        activ = activations['value'][0]  # C,H,W
        for i in range(activ.shape[0]):
            activ[i, :, :] *= pooled_grads[i]
        cam = torch.sum(activ, dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        # reconstruct rgb image
        inp = input_tensor.detach().cpu()[0]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb_img = inp * std + mean
        rgb_img = np.transpose(rgb_img.numpy(), (1, 2, 0))
        rgb_img = np.clip(rgb_img, 0, 1)
        overlay = _apply_colormap_on_image(rgb_img, cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0])))

        handle_f.remove(); handle_b.remove()
        return cam, overlay

