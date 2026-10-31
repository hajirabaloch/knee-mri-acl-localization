import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

st.set_page_config(
    page_title="Knee MRI ACL Localization",
    page_icon="🦵",
    layout="wide"
)

st.title("🦵 Knee MRI ACL Region Localization")
st.write(
    "Upload a knee MRI image to identify the approximate "
    "ACL region using a fine-tuned SegFormer model."
)

MODEL_ID = "AjraBaluch488/knee-mri-acl-localization"

@st.cache_resource
def load_model():
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device

try:
    with st.spinner("Loading AI model..."):
        model, device = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a knee MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize((256, 256), Image.Resampling.BILINEAR)

    image_array = np.array(resized).astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_array).permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with st.spinner("Analyzing MRI image..."):
        with torch.no_grad():
            outputs = model(pixel_values=image_tensor)
            logits = F.interpolate(
                outputs.logits,
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            )
            prediction = torch.argmax(logits, dim=1)[0].cpu().numpy()

    mask = (prediction == 1).astype(np.uint8) * 255

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original MRI")
        st.image(resized, use_container_width=True)

    with col2:
        st.subheader("Predicted ACL Region")
        st.image(mask, clamp=True, use_container_width=True)

    st.success("MRI image analysis completed!")

    st.caption(
        "Research demonstration for approximate ACL region localization. "
        "This tool is not intended for medical diagnosis."
    )
