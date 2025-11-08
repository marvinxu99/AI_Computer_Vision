#
# At the command line: streamlit run strealit_app.py
#

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import pandas as pd


# -------------------------------
# 1. Page setup
# -------------------------------
st.set_page_config(page_title="KinSight 👁️", layout="centered")
st.title("👁️ KinSight — Family Image Classifier")
st.markdown("A ResNet-18–based neural network that identifies household members (Eric, Jane, Marvin, Tobin, and Wesley 🐶).")

# -------------------------------
# 2. Model setup
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "D:/dDev/AI_Computer_Vision/Projects/KinSight/model_20251107_160614_0_saved"
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

model = load_model()
classes = ('Eric', 'Jane', 'Marvin', 'Tobin', 'Wesley')

# -------------------------------
# 3. Image transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image: Image.Image):
    """Run inference on a PIL image and return prediction + confidence."""
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        output = F.softmax(output, dim=1)
        score, idx = torch.max(output, 1)
    label = classes[idx.item()]
    return label, score.item()


# -------------------------------
# 4. Webcam or upload input
# -------------------------------
st.sidebar.header("Choose Input Source")
option = st.sidebar.radio("Select image source:", ["📷 Webcam", "📁 Upload Image"])

# Webcam mode
if option == "📷 Webcam":
    st.info("Click 'Capture' below to take a snapshot from your webcam.")
    camera_image = st.camera_input("Take a picture")

    if camera_image:
        image = Image.open(BytesIO(camera_image.getvalue())).convert('RGB')
        st.image(image, caption="Captured Image", width='stretch')
        label, score = predict_image(image)
        st.success(f"**Prediction:** {label} ({score*100:.2f}% confidence)")

# File upload mode
elif option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", width='stretch')
        label, score = predict_image(image)
        st.success(f"**Prediction:** {label} ({score*100:.2f}% confidence)")

# -------------------------------
# 5. Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by **Marvin Xu**, Clinical Informatics Specialist & Developer  \n[Vancouver, BC, Canada](https://www.mxfhir.com)")
