import os
import streamlit as st
import tensorflow as tf
# If you are loading PKL files for PCA/Scaler/VQC:
import pickle  # or joblib, depending on your setup

# -----------------------------------------------------------------------------
# 1. SETUP BASE PATHS
# -----------------------------------------------------------------------------
# Resolves the directory where app.py lives on Streamlit Cloud or locally
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# -----------------------------------------------------------------------------
# 2. LOAD MODELS FUNCTION
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """
    Loads CNN, PCA, Scaler, VQC weights, and Metadata with path safety checks.
    """
    cnn_path = os.path.join(MODELS_DIR, "cnn_model.keras")
    
    # Debugging Check: Ensure the file actually exists before TensorFlow tries to load it
    if not os.path.exists(cnn_path):
        # List files in the models directory to help debug
        available_files = os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else "Directory 'models' does not exist"
        st.error(
            f"❌ **Model File Not Found!**\n\n"
            f"Expected path: `{cnn_path}`\n\n"
            f"Contents of `models/` directory: `{available_files}`\n\n"
            f"**Action Required:** Ensure `cnn_model.keras` is committed to GitHub and pushed to your repo."
        )
        st.stop()

    # Load CNN Model
    try:
        cnn = tf.keras.models.load_model(cnn_path, compile=False)
    except Exception as e:
        st.error(f"❌ **Failed to load CNN model:** {e}")
        st.stop()

    # -------------------------------------------------------------------------
    # Update these paths and loading methods to match your actual model files!
    # -------------------------------------------------------------------------
    try:
        with open(os.path.join(MODELS_DIR, "pca.pkl"), "rb") as f:
            pca = pickle.load(f)

        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        with open(os.path.join(MODELS_DIR, "vqc_weights.pkl"), "rb") as f:
            vqc_weights = pickle.load(f)

        with open(os.path.join(MODELS_DIR, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

    except Exception as e:
        st.warning(f"⚠️ Warning loading supporting files (PCA/Scaler/VQC/Metadata): {e}")
        # Assign fallbacks or handle as needed for your specific app structure
        pca, scaler, vqc_weights, metadata = None, None, None, None

    return cnn, pca, scaler, vqc_weights, metadata


# -----------------------------------------------------------------------------
# 3. CALL FUNCTION IN MAIN FLOW
# -----------------------------------------------------------------------------
cnn, pca, scaler, vqc_weights, metadata = load_models()
