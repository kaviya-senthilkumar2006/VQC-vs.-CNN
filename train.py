import os
import json
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import pennylane as qml


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="VQC vs CNN",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 VQC vs CNN")
st.write(
    "Compare a Classical CNN with a Variational Quantum Classifier."
)


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

CNN_PATH = os.path.join(
    MODELS_DIR,
    "cnn_model.keras"
)

PCA_PATH = os.path.join(
    MODELS_DIR,
    "pca.joblib"
)

SCALER_PATH = os.path.join(
    MODELS_DIR,
    "scaler.joblib"
)

VQC_PATH = os.path.join(
    MODELS_DIR,
    "vqc_weights.npy"
)

METADATA_PATH = os.path.join(
    MODELS_DIR,
    "metadata.json"
)


# ============================================================
# CHECK MODELS FOLDER
# ============================================================

if not os.path.isdir(MODELS_DIR):

    st.error("❌ 'models' folder was not found.")

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

        missing_files.append(
            (name, path)
        )


if missing_files:

    st.error(
        "❌ Some model files are missing."
    )

    for name, path in missing_files:

        st.write(
            f"**{name}:**"
        )

        st.code(path)

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    cnn = tf.keras.models.load_model(
        CNN_PATH
    )

    pca = joblib.load(
        PCA_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    vqc_weights = np.load(
        VQC_PATH
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    return (
        cnn,
        pca,
        scaler,
        vqc_weights,
        metadata
    )


# ============================================================
# LOAD ALL MODELS
# ============================================================

try:

    (
        cnn,
        pca,
        scaler,
        vqc_weights,
        metadata
    ) = load_models()

    st.success(
        "✅ All models loaded successfully."
    )

except Exception as e:

    st.error(
        "❌ Error while loading models."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VQC SETUP
# ============================================================

N_QUBITS = int(
    metadata.get(
        "n_qubits",
        4
    )
)

N_LAYERS = int(
    metadata.get(
        "n_layers",
        2
    )
)


# ============================================================
# QUANTUM DEVICE
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# VQC CIRCUIT
# ============================================================

@qml.qnode(
    dev,
    interface="autograd"
)
def vqc_circuit(
    x,
    weights
):

    # Same embedding used during training
    qml.AngleEmbedding(
        x,
        wires=range(N_QUBITS),
        rotation="Y"
    )

    # Same variational layers used during training
    qml.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS)
    )

    # Same measurement used during training
    return qml.expval(
        qml.PauliZ(0)
    )


# ============================================================
# VQC PROBABILITY
# ============================================================

def vqc_probability(
    x,
    weights
):

    result = vqc_circuit(
        x,
        weights
    )

    return (
        result + 1.0
    ) / 2.0


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader(
    "📦 Model Information"
)

col1, col2 = st.columns(2)


with col1:

    st.write("CNN: ✅ Loaded")
    st.write("PCA: ✅ Loaded")
    st.write("Scaler: ✅ Loaded")


with col2:

    st.write("VQC: ✅ Loaded")

    st.write(
        f"Qubits: **{N_QUBITS}**"
    )

    st.write(
        f"Layers: **{N_LAYERS}**"
    )


# ============================================================
# METADATA
# ============================================================

with st.expander(
    "🔎 View Metadata"
):

    st.json(metadata)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader(
    "🖼️ Upload an Image"
)

uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "jfif"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        # ====================================================
        # 1. OPEN IMAGE
        # ====================================================

        # The training dataset contains grayscale images.
        image = Image.open(
            uploaded_file
        ).convert("L")


        # ====================================================
        # 2. DISPLAY IMAGE
        # ====================================================

        st.image(
            image,
            caption="Uploaded Grayscale Image",
            width=250
        )


        # ====================================================
        # 3. RESIZE TO 8 x 8
        # ====================================================

        image = image.resize(
            (8, 8)
        )


        # ====================================================
        # 4. CONVERT TO NUMPY
        # ====================================================

        image_array = np.array(
            image
        ).astype(
            "float32"
        )


        # ====================================================
        # 5. NORMALIZE
        # ====================================================

        image_array = (
            image_array / 255.0
        )


        # ====================================================
        # 6. CNN INPUT
        # ====================================================

        cnn_input = np.expand_dims(
            image_array,
            axis=-1
        )

        cnn_input = np.expand_dims(
            cnn_input,
            axis=0
        )


        st.write(
            "CNN input shape:"
        )

        st.code(
            str(cnn_input.shape)
        )


        # ====================================================
        # 7. CNN PREDICTION
        # ====================================================

        cnn_output = cnn.predict(
            cnn_input,
            verbose=0
        )


        cnn_probability = float(
            cnn_output[0][0]
        )


        if cnn_probability >= 0.5:

            cnn_class = 1

            cnn_confidence = (
                cnn_probability
            )

        else:

            cnn_class = 0

            cnn_confidence = (
                1.0 - cnn_probability
            )


        # ====================================================
        # 8. PREPARE IMAGE FOR VQC
        # ====================================================

        # Flatten 8x8 image
        flat_image = image_array.reshape(
            1,
            -1
        )


        # ====================================================
        # 9. PCA
        # ====================================================

        pca_features = pca.transform(
            flat_image
        )


        # ====================================================
        # 10. STANDARD SCALER
        # ====================================================

        vqc_input = scaler.transform(
            pca_features
        )


        # ====================================================
        # 11. VQC PREDICTION
        # ====================================================

        vqc_probability_value = float(
            vqc_probability(
                vqc_input[0],
                vqc_weights
            )
        )


        # ====================================================
        # 12. VQC CLASS
        # ====================================================

        if vqc_probability_value >= 0.5:

            vqc_class = 1

            vqc_confidence = (
                vqc_probability_value
            )

        else:

            vqc_class = 0

            vqc_confidence = (
                1.0 -
                vqc_probability_value
            )


        # ====================================================
        # 13. DISPLAY CNN RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "🧠 CNN Prediction"
        )

        cnn_col1, cnn_col2 = st.columns(2)


        with cnn_col1:

            st.metric(
                "Predicted Digit",
                str(cnn_class)
            )


        with cnn_col2:

            st.metric(
                "Confidence",
                f"{cnn_confidence * 100:.2f}%"
            )


        # ====================================================
        # 14. DISPLAY VQC RESULT
        # ====================================================

        st.subheader(
            "⚛️ VQC Prediction"
        )

        vqc_col1, vqc_col2 = st.columns(2)


        with vqc_col1:

            st.metric(
                "Predicted Digit",
                str(vqc_class)
            )


        with vqc_col2:

            st.metric(
                "Confidence",
                f"{vqc_confidence * 100:.2f}%"
            )


        # ====================================================
        # 15. COMPARISON
        # ====================================================

        st.divider()

        st.subheader(
            "📊 CNN vs VQC"
        )


        comparison_col1, comparison_col2 = st.columns(2)


        with comparison_col1:

            st.write(
                "### 🧠 CNN"
            )

            st.write(
                f"Prediction: **{cnn_class}**"
            )

            st.write(
                f"Confidence: "
                f"**{cnn_confidence * 100:.2f}%**"
            )


        with comparison_col2:

            st.write(
                "### ⚛️ VQC"
            )

            st.write(
                f"Prediction: **{vqc_class}**"
            )

            st.write(
                f"Confidence: "
                f"**{vqc_confidence * 100:.2f}%**"
            )


        # ====================================================
        # 16. AGREEMENT
        # ====================================================

        st.subheader(
            "🔍 Prediction Comparison"
        )


        if cnn_class == vqc_class:

            st.success(
                f"Both models predicted digit **{cnn_class}**."
            )

        else:

            st.warning(
                "The CNN and VQC produced different predictions."
            )


        # ====================================================
        # 17. VQC DETAILS
        # ====================================================

        with st.expander(
            "⚛️ View VQC Processing Details"
        ):

            st.write(
                "Original image:"
            )

            st.write(
                "8 × 8 = 64 pixels"
            )

            st.write(
                "After PCA:"
            )

            st.write(
                f"{pca_features.shape[1]} features"
            )

            st.write(
                "After StandardScaler:"
            )

            st.write(
                f"{vqc_input.shape[1]} features"
            )

            st.write(
                "Number of qubits:"
            )

            st.write(
                N_QUBITS
            )

            st.write(
                "Number of VQC layers:"
            )

            st.write(
                N_LAYERS
            )

            st.write(
                "VQC probability:"
            )

            st.write(
                f"{vqc_probability_value:.6f}"
            )


    except Exception as e:

        st.error(
            "❌ Error while processing the image."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "VQC vs CNN — Hybrid Quantum-Classical Machine Learning"
)
  
