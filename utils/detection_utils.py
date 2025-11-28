

import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.ops as ops
import torchvision.transforms as T


def load_model(path=None, device='cpu', model_type='torch'):
    """
    Minimal loader. If path is None, return None.
    For ultralytics YOLO you should use: from ultralytics import YOLO; model = YOLO(path)
    For torch .pt/.pth saved model this tries torch.load.
    """
    if path is None:
        return None
    device = torch.device(device)
    try:
        model = torch.load(path, map_location=device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model at {path}. Load model in notebook and pass model object to run_detection. Error: {e}")


def preprocess_for_detection(pil_image, target_size=640):
    """Resize while keeping aspect ratio, return letterboxed image (target_size x target_size) and meta"""
    img = np.array(pil_image.convert("RGB"))
    h0, w0 = img.shape[:2]
    r = target_size / max(h0, w0)
    new_w, new_h = int(w0 * r), int(h0 * r)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dw, dh = (target_size - new_w) // 2, (target_size - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w, :] = resized
    meta = {"orig_shape": (h0, w0), "resized_shape": (new_h, new_w), "pad": (dw, dh), "scale": r}
    return canvas, meta


def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    xmin = x - w / 2
    ymin = y - h / 2
    xmax = x + w / 2
    ymax = y + h / 2
    return [xmin, ymin, xmax, ymax]


def non_max_suppression(boxes, scores, iou_threshold=0.45):
    """boxes: Nx4 numpy (xyxy), scores: N numpy -> returns indices keep"""
    if len(boxes) == 0:
        return []
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = ops.nms(boxes_t, scores_t, iou_threshold).numpy().tolist()
    return keep


def run_detection(model, image, device='cpu', conf_thres=0.25, iou_thres=0.45, class_names=None):
    """
    Run inference on a preprocessed image (H,W,3 uint8). Supports:
    - Ultralytics YOLO (model(image) returns Results with .xyxy or .pred)
    - Torchvision Faster R-CNN (model([tensor])[0] returns dict with boxes, scores, labels)

    Returns list of detections with keys: xmin, ymin, xmax, ymax, score, class_id, class_name
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Try ultralytics / yolov-like API
    try:
        results = model(image)
        # ultralytics often has .xyxy or .pred
        if hasattr(results, 'xyxy'):
            preds = results.xyxy[0].cpu().numpy()
            dets = []
            for *box, conf, cls in preds:
                if conf < conf_thres:
                    continue
                xmin, ymin, xmax, ymax = map(float, box)
                dets.append({
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "score": float(conf),
                    "class_id": int(cls),
                    "class_name": (class_names[int(cls)] if class_names else None)
                })
            return dets
        if hasattr(results, 'pred') and len(results.pred) > 0:
            preds = results.pred[0].cpu().numpy()
            dets = []
            for row in preds:
                xmin, ymin, xmax, ymax, conf, cls = row
                if conf < conf_thres:
                    continue
                dets.append({
                    "xmin": float(xmin),
                    "ymin": float(ymin),
                    "xmax": float(xmax),
                    "ymax": float(ymax),
                    "score": float(conf),
                    "class_id": int(cls),
                    "class_name": (class_names[int(cls)] if class_names else None)
                })
            return dets
    except Exception:
        pass

    # Try torchvision Faster R-CNN style
    try:
        transform = T.Compose([T.ToTensor()])
        img_t = transform(Image.fromarray(image)).to(device)
        with torch.no_grad():
            outputs = model([img_t])[0]
        boxes = outputs.get('boxes', torch.tensor([])).cpu().numpy()
        scores = outputs.get('scores', torch.tensor([])).cpu().numpy()
        labels = outputs.get('labels', torch.tensor([])).cpu().numpy() if 'labels' in outputs else [None] * len(scores)
        dets = []
        for box, score, lab in zip(boxes, scores, labels):
            if score < conf_thres:
                continue
            xmin, ymin, xmax, ymax = box.tolist()
            dets.append({
                "xmin": float(xmin),
                "ymin": float(ymin),
                "xmax": float(xmax),
                "ymax": float(ymax),
                "score": float(score),
                "class_id": int(lab) if lab is not None else None,
                "class_name": (class_names[int(lab)] if (class_names and lab is not None) else None)
            })
        return dets
    except Exception as e:
        raise RuntimeError(f"Failed to run detection with provided model. Error: {e}")


def scale_boxes_back(detections, meta):
    """Convert boxes from letterboxed image back to original image coordinates using meta from preprocess_for_detection"""
    dw, dh = meta['pad']
    r = meta['scale']
    scaled = []
    for d in detections:
        xmin = (d['xmin'] - dw) / r
        ymin = (d['ymin'] - dh) / r
        xmax = (d['xmax'] - dw) / r
        ymax = (d['ymax'] - dh) / r
        scaled.append({**d, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    return scaled


def draw_detections(image, detections, class_names=None, thickness=2):
    """
    Draw detections on an image (numpy HxWx3 uint8). Returns a copy with boxes and labels.
    """
    img = image.copy()
    for det in detections:
        xmin, ymin, xmax, ymax = int(round(det['xmin'])), int(round(det['ymin'])), int(round(det['xmax'])), int(round(det['ymax']))
        score = det.get('score', None)
        cls = det.get('class_id', None)
        label = f"{(class_names[cls] if (class_names and cls is not None) else str(cls))} {score:.2f}" if score is not None else str(cls)
        color = (0, 255, 0)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness=thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (xmin, ymin - th - 6), (xmin + tw, ymin), color, -1)
        cv2.putText(img, label, (xmin, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img
