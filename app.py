import json
from pathlib import Path

import joblib
import numpy as np
import pennylane as qml
import streamlit as st

from PIL import Image, ImageOps

import tensorflow as tf


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "models"


# ---------------------------------------------------------
# STREAMLIT SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="VQC vs CNN",
    page_icon="⚛️",
    layout="centered"
)


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

@st.cache_resource
def load_models():

    cnn = tf.keras.models.load_model(
        MODEL_DIR / "cnn_model.keras"
    )

    pca = joblib.load(
        MODEL_DIR / "pca.joblib"
    )

    scaler = joblib.load(
        MODEL_DIR / "scaler.joblib"
    )

    weights = np.load(
        MODEL_DIR / "vqc_weights.npy"
    )

    with open(
        MODEL_DIR / "metadata.json",
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    return (
        cnn,
        pca,
        scaler,
        weights,
        metadata
    )


cnn, pca, scaler, vqc_weights, metadata = load_models()


# ---------------------------------------------------------
# VQC SETUP
# ---------------------------------------------------------

N_QUBITS = int(
    metadata["n_qubits"]
)

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


@qml.qnode(dev)
def vqc_circuit(
    x,
    weights
):

    qml.AngleEmbedding(
        x,
        wires=range(N_QUBITS),
        rotation="Y"
    )

    qml.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS)
    )

    return qml.expval(
        qml.PauliZ(0)
    )


def vqc_probability(x):

    value = float(
        vqc_circuit(
            x,
            vqc_weights
        )
    )

    return (
        value + 1.0
    ) / 2.0


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------

def preprocess_image(
    uploaded_image
):

    image = Image.open(
        uploaded_image
    ).convert("L")

    image = ImageOps.autocontrast(
        image
    )

    image = image.resize(
        (8, 8)
    )

    image_array = (
        np.asarray(image)
        .astype("float32")
        / 255.0
    )

    # Convert to dark background /
    # bright digit format
    if image_array.mean() > 0.5:

        image_array = (
            1.0 - image_array
        )

    return image_array


# ---------------------------------------------------------
# USER INTERFACE
# ---------------------------------------------------------

st.title("⚛️ VQC vs CNN")

st.write(
    "Compare a Variational Quantum Classifier "
    "(VQC) with a classical Convolutional "
    "Neural Network (CNN)."
)


st.info(
    "This demonstration is trained on "
    "handwritten digits 0 and 1. "
    "Upload an image containing one of "
    "these digits."
)


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------

uploaded = st.file_uploader(
    "Upload a digit image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


if uploaded is not None:

    # ---------------------------------------------
    # PREPROCESS
    # ---------------------------------------------

    image_array = preprocess_image(
        uploaded
    )


    # ---------------------------------------------
    # DISPLAY IMAGE
    # ---------------------------------------------

    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image_array,
        width=180
    )


    # ---------------------------------------------
    # CNN PREDICTION
    # ---------------------------------------------

    cnn_input = (
        image_array[
            np.newaxis,
            ...,
            np.newaxis
        ]
    )

    cnn_probability = float(
        cnn.predict(
            cnn_input,
            verbose=0
        )[0][0]
    )

    cnn_prediction = (
        1
        if cnn_probability >= 0.5
        else 0
    )


    # ---------------------------------------------
    # VQC PREDICTION
    # ---------------------------------------------

    flat_image = (
        image_array
        .reshape(1, -1)
    )

    pca_features = pca.transform(
        flat_image
    )

    vqc_features = scaler.transform(
        pca_features
    )[0]

    vqc_probability_value = (
        vqc_probability(
            vqc_features
        )
    )

    vqc_prediction = (
        1
        if vqc_probability_value >= 0.5
        else 0
    )


    # ---------------------------------------------
    # RESULTS
    # ---------------------------------------------

    st.subheader(
        "Prediction Results"
    )

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "CNN Prediction",
            str(cnn_prediction)
        )

        st.write(
            "Probability of digit 1:",
            f"{cnn_probability:.3f}"
        )


    with col2:

        st.metric(
            "VQC Prediction",
            str(vqc_prediction)
        )

        st.write(
            "Probability of digit 1:",
            f"{vqc_probability_value:.3f}"
        )


    # ---------------------------------------------
    # MODEL ACCURACY
    # ---------------------------------------------

    st.subheader(
        "Model Accuracy"
    )

    st.write(
        "CNN Test Accuracy:",
        f"{metadata['cnn_test_accuracy'] * 100:.2f}%"
    )

    st.write(
        "VQC Test Accuracy:",
        f"{metadata['vqc_test_accuracy'] * 100:.2f}%"
    )


    # ---------------------------------------------
    # COMPARISON
    # ---------------------------------------------

    st.subheader(
        "Model Comparison"
    )

    if cnn_prediction == vqc_prediction:

        st.success(
            f"Both models predicted digit "
            f"**{cnn_prediction}**."
        )

    else:

        st.warning(
            f"The models produced different "
            f"predictions.\n\n"
            f"CNN → **{cnn_prediction}**\n\n"
            f"VQC → **{vqc_prediction}**"
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "VQC = Variational Quantum Classifier | "
    "CNN = Convolutional Neural Network"
)
