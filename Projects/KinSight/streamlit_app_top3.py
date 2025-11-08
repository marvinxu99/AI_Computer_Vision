#
# At the command line: streamlit run strealit_app_top3.py
#

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from io import BytesIO

# -------------------------------
# 1. Page setup
# -------------------------------
st.set_page_config(page_title="KinSight 👁️", layout="centered")
st.title("👁️ KinSight — Family Image Classifier")
st.write("Identify your household members (Eric, Jane, Marvin, Tobin, and Wesley 🐶).")

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

# model-loading feedback
with st.spinner("Loading model..."):
    model = load_model()
st.success("Model loaded successfully ✅")


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

def predict_top3(image: Image.Image):
    """Return top-3 predictions (class, probability)."""
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).squeeze().numpy()
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(classes[i], probs[i]) for i in top3_idx]
    return top3

# -------------------------------
# 4. Sidebar input selection
# -------------------------------
st.sidebar.header("Choose Input Source")
option = st.sidebar.radio("Select image source:", ["📁 Upload Image", "📷 Webcam"])

# -------------------------------
# 5. Upload mode
# -------------------------------
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", width='stretch')
        top3 = predict_top3(image)
        best_label, best_conf = top3[0]
        st.success(f"**Prediction:** {best_label} ({best_conf*100:.2f}% confidence)")

        # Display top-3 chart
        df = pd.DataFrame(top3, columns=["Class", "Confidence"])
        st.subheader("Top-3 Predictions")
        st.bar_chart(df.set_index("Class"))

# -------------------------------
# 6. Webcam mode
# -------------------------------
elif option == "📷 Webcam":
    st.info("Click 'Capture' to take a snapshot from your webcam.")
    camera_image = st.camera_input("Take a picture")

    if camera_image:
        image = Image.open(BytesIO(camera_image.getvalue())).convert('RGB')
        st.image(image, caption="Captured Image", width='stretch')
        top3 = predict_top3(image)
        best_label, best_conf = top3[0]
        st.success(f"**Prediction:** {best_label} ({best_conf*100:.2f}% confidence)")

        # Display top-3 chart
        df = pd.DataFrame(top3, columns=["Class", "Confidence"])
        st.subheader("Top-3 Predictions")
        st.bar_chart(df.set_index("Class"))

# -------------------------------
# 7. Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by **Marvin Xu**, Clinical Informatics Specialist & Developer  \n[Vancouver, BC, Canada](https://www.mxfhir.com)")
