import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# settings
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


CLASSES = ["Acute Otitis Media", "Normal", "Cerumen", "OME"]
BEST_WEIGHTS = "best.pt"


# helpers
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

def get_last_conv_layer(module: nn.Module):
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
    return last_conv

# transforms
eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_mobilenet_v3_large(num_classes=len(CLASSES)).to(device)
state = torch.load(BEST_WEIGHTS, map_location=device)
model.load_state_dict(state)
model.eval()

# UI
st.set_page_config(page_title="Otoscopic Classifier + Grad-CAM", layout="centered")
st.title("Otoscopic Classifier (MobileNetV3-Large) + Grad-CAM")
st.caption("Upload an eardrum image to get the predicted class and an explanation heatmap.")

uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original", use_container_width=True)

    x = eval_tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        pred_prob = float(probs[pred_idx])

    st.markdown(f"Prediction: **{pred_class}**  |  Confidence: **{pred_prob:.3f}**")

    # Grad-CAM on predicted class
    for p in model.parameters():
        if isinstance(p, torch.Tensor):
            p.requires_grad = True
    target_layer = get_last_conv_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]

    # overlay CAM on resized image (0..1)
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    c1, c2 = st.columns(2)
    with c1:
        st.image(img_np, caption="Preprocessed (224x224)", use_container_width=True)
    with c2:
        st.image(cam_image, caption=f"Grad-CAM → {pred_class}", use_container_width=True)

    st.subheader("Class probabilities")
    st.dataframe({"class": CLASSES, "probability": [float(p) for p in probs]}, use_container_width=True)
else:
    st.info("Upload a JPG/PNG to begin.")
