#
# At the command line: streamlit run strealit_app_realtime.py
#

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import time

# -------------------------------
# 1. Page setup
# -------------------------------
st.set_page_config(page_title="KinSight Live 👁️", layout="centered")
st.title("👁️ KinSight — Live Family Recognition")
st.write("Real-time webcam classifier identifying Eric, Jane, Marvin, Tobin, and Wesley 🐶.")

# -------------------------------
# 2. Load model (cached)
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
# 3. Define preprocessing transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_top3(frame):
    """Run inference on a single OpenCV frame and return top-3 predictions."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).squeeze().numpy()
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(classes[i], probs[i]) for i in top3_idx]
    return top3, probs

# -------------------------------
# 4. Streamlit UI controls
# -------------------------------
st.sidebar.header("⚙️ Settings")
run = st.sidebar.checkbox("Start Webcam", value=False)
frame_window = st.empty()
chart_placeholder = st.empty()

# -------------------------------
# 5. Live camera loop
# -------------------------------
camera = cv2.VideoCapture(0)  # 0 = default webcam
if not camera.isOpened():
    st.error("❌ Could not access the webcam. Please check your camera connection.")
else:
    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("⚠️ Failed to read from webcam.")
            break

        # Predict top-3
        top3, probs = predict_top3(frame)
        best_label, best_conf = top3[0]

        # Overlay text on frame
        text = f"{best_label} ({best_conf*100:.1f}%)"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert BGR → RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update live frame
        frame_window.image(frame_rgb, caption=f"Prediction: {text}", width='stretch')

        # Update live top-3 confidence chart
        df = pd.DataFrame(top3, columns=["Class", "Confidence"])
        chart_placeholder.bar_chart(df.set_index("Class"))

        # Control frame rate
        time.sleep(0.1)

    camera.release()

# -------------------------------
# 6. Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by **Marvin Xu**, Clinical Informatics Specialist & Developer  \n[Vancouver, BC, Canada](https://www.mxfhir.com)")
