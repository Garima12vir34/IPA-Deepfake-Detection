import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from model import load_model
from utils import predict_image
from gradcam import create_gradcam

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "baseline.pth"
METRICS_PATH = PROJECT_DIR / "metrics.json"

st.title("🔍 AI-Powered Deepfake Detection")
st.write(
    "Upload a face image to check whether it is classified as Real or Fake. "
    "The Grad-CAM visualization highlights image regions that influenced the prediction."
)

st.info("Project class mapping: Fake = 0, Real = 1. The model outputs the probability of Real.")

@st.cache_resource
def get_model(model_path_string):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path_string, device)
    return model, device

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Ask Member B for baseline.pth and place it in "
        "the project's models/ folder."
    )
    st.stop()

try:
    model, device = get_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

st.caption(f"Running on: {device}")

uploaded_file = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        result = predict_image(model, image, device)

        left, right = st.columns(2)

        with left:
            st.subheader("Uploaded image")
            st.image(image, use_container_width=True)

        with right:
            st.subheader("Prediction")
            if result["label"] == "Real":
                st.success("Prediction: REAL")
            else:
                st.error("Prediction: FAKE")

            st.metric(
                "Confidence in predicted class",
                f"{max(result['real_probability'], result['fake_probability']) * 100:.2f}%"
            )
            st.write(f"Real probability: **{result['real_probability'] * 100:.2f}%**")
            st.write(f"Fake probability: **{result['fake_probability'] * 100:.2f}%**")

        st.subheader("Grad-CAM Explainability")
        st.write(
            "Highlighted areas show regions that contributed to the selected class score. "
            "They are an explanation aid, not proof that an image is fake or real."
        )

        with st.spinner("Generating Grad-CAM heatmap..."):
            heatmap = create_gradcam(
                model,
                result["input_tensor"],
                image,
                result["label"]
            )

        st.image(heatmap, caption="Grad-CAM heatmap overlay", use_container_width=True)

    except Exception as error:
        st.error(f"Unable to process this image: {error}")

with st.expander("About the project"):
    st.markdown("""
    - **Model:** ResNet18 with a single output neuron
    - **Input size:** 128 × 128 pixels
    - **Preprocessing:** ImageNet normalization
    - **Explainability:** Grad-CAM on the final ResNet18 convolutional block
    - **Classes:** Fake (0), Real (1)
    """)

if METRICS_PATH.exists():
    with st.expander("Model evaluation metrics"):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as file:
                metrics = json.load(file)
            st.json(metrics)
        except Exception as error:
            st.warning(f"Could not read metrics.json: {error}")

st.divider()
st.caption(
    "Academic project demonstration. Predictions may be affected by dataset bias, "
    "image compression, and unseen deepfake generation methods."
)
