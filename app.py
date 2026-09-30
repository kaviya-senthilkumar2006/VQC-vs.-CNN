import os
import json
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="VQC vs CNN",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 VQC vs CNN")
st.write("Compare Classical CNN and Variational Quantum Classifier models.")


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CNN_PATH = os.path.join(MODELS_DIR, "cnn_model.keras")
PCA_PATH = os.path.join(MODELS_DIR, "pca.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
VQC_PATH = os.path.join(MODELS_DIR, "vqc_weights.npy")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")


# ============================================================
# CHECK MODELS FOLDER
# ============================================================

if not os.path.isdir(MODELS_DIR):

    st.error("❌ 'models' folder was not found.")

    st.write("Expected folder:")
    st.code(MODELS_DIR)

    st.stop()


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = {
    "CNN model": CNN_PATH,
    "PCA": PCA_PATH,
    "Scaler": SCALER_PATH,
    "VQC weights": VQC_PATH,
    "Metadata": METADATA_PATH
}

missing_files = []

for name, path in required_files.items():

    if not os.path.isfile(path):
        missing_files.append((name, path))


if missing_files:

    st.error("❌ Some model files are missing.")

    for name, path in missing_files:
        st.write(f"**{name}:**")
        st.code(path)

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
    vqc_weights = np.load(VQC_PATH)

    # Metadata
    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    return cnn, pca, scaler, vqc_weights, metadata


# ============================================================
# LOAD ALL FILES
# ============================================================

try:

    cnn, pca, scaler, vqc_weights, metadata = load_models()

    st.success("✅ All model files loaded successfully.")

except Exception as e:

    st.error("❌ Error while loading the models.")

    st.exception(e)

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("📦 Model Information")

col1, col2 = st.columns(2)

with col1:

    st.write("CNN: ✅ Loaded")
    st.write("PCA: ✅ Loaded")
    st.write("Scaler: ✅ Loaded")


with col2:

    st.write("VQC weights: ✅ Loaded")
    st.write("Metadata: ✅ Loaded")


# ============================================================
# SHOW CNN INPUT SHAPE
# ============================================================

st.subheader("🧠 CNN Input Information")

input_shape = cnn.input_shape

if isinstance(input_shape, list):
    input_shape = input_shape[0]

st.write("Model input shape:")

st.code(str(input_shape))


# ============================================================
# METADATA
# ============================================================

with st.expander("🔎 View Metadata"):

    st.json(metadata)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("🖼️ Upload an Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "jfif"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        # IMPORTANT:
        # The CNN expects 1 channel.
        # Therefore we convert the image to grayscale.

        image = Image.open(uploaded_file).convert("L")


        # ----------------------------------------------------
        # DISPLAY IMAGE
        # ----------------------------------------------------

        st.image(
            image,
            caption="Uploaded Grayscale Image",
            use_container_width=True
        )


        # ----------------------------------------------------
        # GET MODEL INPUT SIZE
        # ----------------------------------------------------

        input_shape = cnn.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]


        height = input_shape[1]
        width = input_shape[2]


        # ----------------------------------------------------
        # RESIZE IMAGE
        # ----------------------------------------------------

        image = image.resize(
            (width, height)
        )


        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        image_array = np.array(
            image
        ).astype("float32")


        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        image_array = image_array / 255.0


        # ----------------------------------------------------
        # ADD CHANNEL DIMENSION
        # ----------------------------------------------------

        # Before:
        # (8, 8)

        # After:
        # (8, 8, 1)

        image_array = np.expand_dims(
            image_array,
            axis=-1
        )


        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        # ----------------------------------------------------

        # Before:
        # (8, 8, 1)

        # After:
        # (1, 8, 8, 1)

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # SHOW FINAL INPUT SHAPE
        # ----------------------------------------------------

        st.write("Image input shape sent to CNN:")

        st.code(str(image_array.shape))


        # ----------------------------------------------------
        # CNN PREDICTION
        # ----------------------------------------------------

        prediction = cnn.predict(
            image_array,
            verbose=0
        )


        # ----------------------------------------------------
        # DETERMINE PREDICTION
        # ----------------------------------------------------

        if prediction.shape[-1] == 1:

            probability = float(
                prediction[0][0]
            )

            if probability >= 0.5:

                predicted_class = 1
                confidence = probability

            else:

                predicted_class = 0
                confidence = 1.0 - probability


        else:

            predicted_class = int(
                np.argmax(prediction[0])
            )

            confidence = float(
                np.max(prediction[0])
            )


        # ----------------------------------------------------
        # GET CLASS NAMES
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
        # GET PREDICTION NAME
        # ----------------------------------------------------

        if class_names is not None:

            if predicted_class < len(class_names):

                predicted_name = class_names[
                    predicted_class
                ]

            else:

                predicted_name = str(
                    predicted_class
                )

        else:

            predicted_name = str(
                predicted_class
            )


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.subheader("🎯 CNN Prediction")

        st.success(
            f"Prediction: **{predicted_name}**"
        )

        st.write(
            f"Class number: **{predicted_class}**"
        )

        st.write(
            f"Confidence: **{confidence * 100:.2f}%**"
        )

        st.progress(
            min(
                max(confidence, 0.0),
                1.0
            )
        )


        # ----------------------------------------------------
        # RAW PREDICTION
        # ----------------------------------------------------

        with st.expander("🔬 View Raw CNN Output"):

            st.write(prediction)


    except Exception as e:

        st.error(
            "❌ Error while processing the image."
        )

        st.exception(e)


# ============================================================
# VQC INFORMATION
# ============================================================

st.divider()

st.subheader("⚛️ VQC Information")

st.write(
    f"VQC weights loaded: "
    f"**{vqc_weights.size} parameters**"
)


# ============================================================
# PCA / SCALER INFORMATION
# ============================================================

st.subheader("📊 Supporting Components")

col1, col2 = st.columns(2)

with col1:

    st.write("PCA: ✅ Loaded")

    try:
        st.write(
            f"Components: **{pca.n_components_}**"
        )
    except Exception:
        st.write("PCA information available")


with col2:

    st.write("Scaler: ✅ Loaded")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "VQC vs CNN — Hybrid Quantum-Classical Machine Learning"
)

