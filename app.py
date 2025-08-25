import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Large_Weights
from PIL import Image, ImageOps
import numpy as np
import io

# Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(page_title="Otoscopy Classifier (MobileNetV3 + Grad-CAM)", page_icon="🩺", layout="wide")

# ---------- Constants ----------
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ---------- Helpers ----------
def get_last_conv_layer(module: nn.Module):
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
    return last_conv

def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    # Replace classifier final layer
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, num_classes)
    # Ensure gradients are enabled for Grad-CAM
    for p in model.parameters():
        p.requires_grad = True
    return model

@st.cache_resource(show_spinner=False)
def load_model_from_bytes(num_classes: int, weight_bytes: bytes, device: str = "cpu") -> nn.Module:
    model = build_model(num_classes)
    state = torch.load(io.BytesIO(weight_bytes), map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def default_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

def pil_to_model_input(pil_img: Image.Image, tfm, device: str = "cpu") -> torch.Tensor:
    # Force RGB
    img = pil_img.convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    return x

def predict(model: nn.Module, x: torch.Tensor):
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        top_idx = int(np.argmax(probs))
    return probs, top_idx

def make_gradcam(model: nn.Module, x: torch.Tensor, pred_idx: int, use_cuda: bool = False):
    target_layer = get_last_conv_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=use_cuda)
    grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]
    return grayscale_cam

def overlay_cam_on_pil(pil_img: Image.Image, grayscale_cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    # Convert PIL -> np (0..1), then overlay CAM using grad-cam helper
    img_resized = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    out = Image.fromarray(cam_image)
    if alpha != 1.0:
        # Blend original and CAM for nicer look
        out = Image.blend(img_resized, out, alpha=alpha)
    return out

def parse_classes_input(text: str):
    # Split by comma, strip whitespace, drop empties
    names = [t.strip() for t in text.split(",")]
    names = [t for t in names if len(t) > 0]
    # Fallback
    if not names:
        names = [f"class_{i}" for i in range(5)]
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for n in names:
        if n not in seen:
            ordered.append(n)
            seen.add(n)
    return ordered

# ---------- Sidebar ----------
st.sidebar.header("Setup")

# Classes list
default_classes_text = "AOM, OME, Normal, TM_Perforation, Cerumen"
classes_text = st.sidebar.text_input("Class names (comma-separated, order must match training)", value=default_classes_text)
class_names = parse_classes_input(classes_text)
num_classes = len(class_names)
st.sidebar.write(f"Detected {num_classes} classes: {class_names}")

alpha = st.sidebar.slider("Grad-CAM overlay blend", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# Weights uploader (optional). If not provided, try to read local best.pt.
weights_file = st.sidebar.file_uploader("Upload model weights (.pt)", type=["pt"])
weight_bytes = None

if weights_file is not None:
    weight_bytes = weights_file.read()
elif os.path.exists("best.pt"):
    with open("best.pt", "rb") as f:
        weight_bytes = f.read()

if weight_bytes is None:
    st.sidebar.warning("No weights provided. Upload a .pt file or place best.pt next to app.py.")
else:
    st.sidebar.success("Weights ready.")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device: {device}")

# ---------- Main ----------
st.title("Otoscopy Classifier (MobileNetV3 + Grad-CAM)")
st.caption("Upload a tympanic membrane image, get the predicted class and a Grad-CAM heatmap highlighting discriminative regions.")

uploaded = st.file_uploader("Upload otoscopic image (PNG/JPG)", type=["png","jpg","jpeg"])

# Lazy-load model once weights are present
model = None
if weight_bytes is not None:
    try:
        model = load_model_from_bytes(num_classes, weight_bytes, device=device)
    except Exception as e:
        st.error(f"Failed to load model with {num_classes} classes: {e}")
        model = None

if uploaded is not None:
    try:
        pil_img = Image.open(uploaded)
    except Exception as e:
        st.error(f"Could not open image: {e}")
        pil_img = None

    if pil_img is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(pil_img, caption="Uploaded image", use_column_width=True)

        if model is None:
            st.warning("Model is not loaded yet. Provide weights in the sidebar.")
        else:
            tfm = default_transform()
            x = pil_to_model_input(pil_img, tfm, device=device)

            # Predict
            probs, top_idx = predict(model, x)
            pred_label = class_names[top_idx] if top_idx < len(class_names) else f"class_{top_idx}"
            conf = float(probs[top_idx])

            # Grad-CAM
            try:
                grayscale_cam = make_gradcam(model, x, pred_idx=top_idx, use_cuda=(device == "cuda"))
                cam_pil = overlay_cam_on_pil(pil_img, grayscale_cam, alpha=alpha)
            except Exception as e:
                grayscale_cam = None
                cam_pil = None
                st.error(f"Grad-CAM generation failed: {e}")

            # Display results
            with col2:
                st.subheader("Prediction")
                st.metric(label="Top class", value=pred_label, delta=f"{conf*100:.1f}%")

                # Table of probabilities
                prob_rows = []
                for idx, name in enumerate(class_names):
                    p = float(probs[idx]) if idx < len(probs) else 0.0
                    prob_rows.append({"class": name, "probability": p})
                # Sort by probability desc
                prob_rows = sorted(prob_rows, key=lambda r: r["probability"], reverse=True)
                st.dataframe(prob_rows, hide_index=True, use_container_width=True)

                if cam_pil is not None:
                    st.subheader("Grad-CAM")
                    st.image(cam_pil, caption="Heatmap overlay", use_column_width=True)

        st.info("Tip: make sure your class list matches the order used during training, and that your weights were trained for the same number of classes.")

# Footer
st.caption("MobileNetV3-Large (ImageNet init) with Grad-CAM. Normalisation: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]. Image is resized to 224x224.")