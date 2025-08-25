import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
from torchvision import models, transforms
from matplotlib import cm  # for heatmap coloring

# -------- settings --------
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Put your classes in EXACT order from training
CLASSES = ["Normal","AOM","Otitis externa","Cerumen","Perforation"]

# If best.pt is in the same folder as app.py, keep this:
BEST_WEIGHTS = "best.pt"

# -------- helpers (same as your training notebook) --------
def replace_classifier_mnv3(model, num_classes):
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, num_classes)
    return model

def get_mobilenet_v3_large(num_classes):
    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )
    for p in model.parameters():
        if isinstance(p, torch.Tensor):
            p.requires_grad = False
    model = replace_classifier_mnv3(model, num_classes)
    return model

def get_last_conv_name(model: nn.Module) -> str:
    # TorchCAM needs the *name* of the last Conv2d
    last_name = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last_name = name
    if last_name is None:
        raise RuntimeError("No Conv2d found for Grad-CAM.")
    return last_name

# -------- transforms --------
eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# -------- load model --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_mobilenet_v3_large(num_classes=len(CLASSES)).to(device)
state = torch.load(BEST_WEIGHTS, map_location=device)
model.load_state_dict(state, strict=True)
model.eval()

# -------- TorchCAM (no OpenCV) --------
from torchcam.methods import GradCAM

target_layer_name = get_last_conv_name(model)
cam_extractor = GradCAM(model, target_layer=target_layer_name)

def make_cam_overlay(rgb_float01: np.ndarray, cam_float01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    rgb_float01: (H,W,3) in [0,1]
    cam_float01: (H,W) in [0,1]
    returns (H,W,3) uint8 overlay
    """
    # colorize heatmap using matplotlib (jet)
    heatmap = cm.get_cmap("jet")(cam_float01)[..., :3]  # (H,W,3) in [0,1]
    overlay = (1 - alpha) * rgb_float01 + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)

# -------- UI --------
st.set_page_config(page_title="Otoscopic Classifier + Grad-CAM", layout="centered")
st.title("Otoscopic Classifier (MobileNetV3-Large) + Grad-CAM")
st.caption("Upload an eardrum image to get the predicted class and a Grad-CAM heatmap (no OpenCV).")

uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original", use_container_width=True)

    # preprocess
    x = eval_tfms(img).unsqueeze(0).to(device)

    # forward + prediction
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    pred_prob = float(probs[pred_idx])

    st.markdown(f"Prediction: **{pred_class}**  |  Confidence: **{pred_prob:.3f}**")

    # TorchCAM: get CAM for predicted class
    # TorchCAM expects a forward pass; we already ran it. Now extract the CAM.
    # If needed, run again with gradients enabled to be safe:
    model.zero_grad(set_to_none=True)
    scores = model(x)
    scores[0, pred_idx].backward(retain_graph=True)
    cams = cam_extractor(pred_idx, scores)  # list of (H,W) tensors per target layer
    cam = cams[0].detach().cpu().numpy()

    # Normalize CAM to [0,1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam_norm = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = np.zeros_like(cam)

    # Prepare resized image array in [0,1]
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.asarray(img_resized).astype(np.float32) / 255.0

    # Create overlay without OpenCV
    cam_overlay = make_cam_overlay(img_np, cam_norm, alpha=0.45)

    c1, c2 = st.columns(2)
    with c1:
        st.image((img_np * 255).astype(np.uint8), caption="Resized (224×224)", use_container_width=True)
    with c2:
        st.image(cam_overlay, caption=f"Grad‑CAM → {pred_class}", use_container_width=True)

    st.subheader("Class probabilities")
    st.dataframe({"class": CLASSES, "probability": [float(p) for p in probs]}, use_container_width=True)

else:
    st.info("Upload a JPG/PNG to begin.")
