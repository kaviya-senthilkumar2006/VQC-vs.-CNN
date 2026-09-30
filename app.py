import os
import json
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VQC vs CNN",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


CNN_PATH = os.path.join(MODELS_DIR, "cnn_model.keras")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")
PCA_PATH = os.path.join(MODELS_DIR, "pca.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
VQC_WEIGHTS_PATH = os.path.join(MODELS_DIR, "vqc_weights.npy")


# ============================================================
# TITLE
# ============================================================

st.title("🤖 VQC vs CNN")
st.write(
    "Compare predictions from a Classical CNN and "
    "a Variational Quantum Classifier (VQC)."
)


# ============================================================
# CHECK MODEL DIRECTORY
# ============================================================

if not os.path.isdir(MODELS_DIR):
    st.error(
        f"❌ Models directory not found.\n\n"
        f"Expected location:\n`{MODELS_DIR}`"
    )
    st.stop()


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = {
    "CNN model": CNN_PATH,
    "Metadata": METADATA_PATH,
    "PCA": PCA_PATH,
    "Scaler": SCALER_PATH,
    "VQC weights": VQC_WEIGHTS_PATH
}

missing_files = []

for name, path in required_files.items():
    if not os.path.isfile(path):
        missing_files.append(f"{name}: `{path}`")


if missing_files:
    st.error("❌ Some required model files are missing:")

    for file in missing_files:
        st.write(file)

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # CNN
    cnn = tf.keras.models.load_model(CNN_PATH)

    # PCA
    pca = joblib.load(PCA_PATH)

    # Scaler
    scaler = joblib.load(SCALER_PATH)

    # VQC weights
    vqc_weights = np.load(VQC_WEIGHTS_PATH)

    # Metadata
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return cnn, pca, scaler, vqc_weights, metadata


# ============================================================
# LOAD
# ============================================================

try:

    cnn, pca, scaler, vqc_weights, metadata = load_models()

    st.success("✅ Models loaded successfully!")

except Exception as e:

    st.error("❌ Error while loading the models.")

    st.code(str(e))

    st.stop()


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

st.subheader("📦 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.write("**CNN:** Loaded ✅")
    st.write("**PCA:** Loaded ✅")
    st.write("**Scaler:** Loaded ✅")

with col2:
    st.write("**VQC weights:** Loaded ✅")
    st.write("**Metadata:** Loaded ✅")


# ============================================================
# DISPLAY METADATA
# ============================================================

with st.expander("🔎 View Metadata"):

    st.json(metadata)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("🖼️ Test Image")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "jfif"]
)


# ============================================================
# IMAGE PREDICTION
# ============================================================

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    try:

        from PIL import Image

        image = Image.open(uploaded_file).convert("RGB")

        # ----------------------------------------------------
        # Determine CNN input size
        # ----------------------------------------------------

        input_shape = cnn.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        height = input_shape[1]
        width = input_shape[2]

        # ----------------------------------------------------
        # Resize image
        # ----------------------------------------------------

        image = image.resize((width, height))

        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        image_array = np.array(image).astype("float32")

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        image_array = image_array / 255.0

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ----------------------------------------------------
        # CNN prediction
        # ----------------------------------------------------

        prediction = cnn.predict(
            image_array,
            verbose=0
        )

        # ----------------------------------------------------
        # Handle different output formats
        # ----------------------------------------------------

        if prediction.shape[-1] == 1:

            probability = float(prediction[0][0])

            predicted_class = (
                1 if probability >= 0.5 else 0
            )

            confidence = (
                probability
                if predicted_class == 1
                else 1 - probability
            )

        else:

            predicted_class = int(
                np.argmax(prediction[0])
            )

            confidence = float(
                np.max(prediction[0])
            )

        # ----------------------------------------------------
        # Get class names
        # ----------------------------------------------------

        class_names = None

        if isinstance(metadata, dict):

            if "class_names" in metadata:
                class_names = metadata["class_names"]

            elif "classes" in metadata:
                class_names = metadata["classes"]

            elif "labels" in metadata:
                class_names = metadata["labels"]


        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        st.subheader("🎯 CNN Prediction")

        if class_names and predicted_class < len(class_names):

            predicted_name = class_names[predicted_class]

        else:

            predicted_name = str(predicted_class)

        st.success(
            f"Prediction: **{predicted_name}**"
        )

        st.write(
            f"Confidence: **{confidence * 100:.2f}%**"
        )

        st.progress(
            min(max(confidence, 0.0), 1.0)
        )


# ============================================================
# VQC INFORMATION
# ============================================================

st.divider()

st.subheader("⚛️ VQC Information")

st.write(
    f"VQC weights loaded successfully."
)

st.write(
    f"Number of VQC parameters: "
    f"**{vqc_weights.size}**"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "VQC vs CNN — Hybrid Quantum-Classical Machine Learning"
)


