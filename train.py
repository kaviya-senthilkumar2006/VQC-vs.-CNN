"""
Train and save both models for the VQC vs CNN project.

Dataset:
    scikit-learn Digits dataset, restricted to digits 0 and 1.

Outputs:
    models/cnn_model.keras
    models/vqc_weights.npy
    models/pca.joblib
    models/scaler.joblib
    models/metadata.json
"""

from pathlib import Path
import json
import numpy as np
import joblib
import pennylane as qml
from pennylane import numpy as pnp

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

digits = load_digits()

X = digits.images.astype("float32") / 16.0
y = digits.target.astype("int64")

# Keep only digits 0 and 1
mask = (y == 0) | (y == 1)

X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ---------------------------------------------------------
# 2. CLASSICAL CNN
# ---------------------------------------------------------

cnn = keras.Sequential([
    layers.Input(shape=(8, 8, 1)),

    layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.Flatten(),

    layers.Dense(
        32,
        activation="relu"
    ),

    layers.Dropout(0.20),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


cnn.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print("\nTraining CNN...")

cnn.fit(
    X_train[..., np.newaxis],
    y_train,
    validation_split=0.15,
    epochs=12,
    batch_size=32,
    verbose=1
)


cnn_loss, cnn_acc = cnn.evaluate(
    X_test[..., np.newaxis],
    y_test,
    verbose=0
)

print("\nCNN test accuracy:", cnn_acc)


# Save CNN
cnn.save(
    MODEL_DIR / "cnn_model.keras"
)


# ---------------------------------------------------------
# 3. PREPARE DATA FOR VQC
# ---------------------------------------------------------

X_train_flat = X_train.reshape(
    len(X_train),
    -1
)

X_test_flat = X_test.reshape(
    len(X_test),
    -1
)


# Reduce 64 pixels to 4 features
pca = PCA(
    n_components=4,
    random_state=SEED
)

X_train_pca = pca.fit_transform(
    X_train_flat
)

X_test_pca = pca.transform(
    X_test_flat
)


# Standardize the four features
scaler = StandardScaler()

X_train_vqc = scaler.fit_transform(
    X_train_pca
)

X_test_vqc = scaler.transform(
    X_test_pca
)


# Save preprocessing objects
joblib.dump(
    pca,
    MODEL_DIR / "pca.joblib"
)

joblib.dump(
    scaler,
    MODEL_DIR / "scaler.joblib"
)


# ---------------------------------------------------------
# 4. VARIATIONAL QUANTUM CLASSIFIER
# ---------------------------------------------------------

N_QUBITS = 4
N_LAYERS = 2

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


@qml.qnode(
    dev,
    interface="autograd"
)
def circuit(x, weights):

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


def vqc_probability(x, weights):

    result = circuit(
        x,
        weights
    )

    return (result + 1.0) / 2.0


def batch_loss(
    weights,
    X_batch,
    y_batch
):

    predictions = pnp.stack([
        vqc_probability(
            x,
            weights
        )
        for x in X_batch
    ])

    return pnp.mean(
        (predictions - y_batch) ** 2
    )


# ---------------------------------------------------------
# INITIALIZE VQC WEIGHTS
# ---------------------------------------------------------

weight_shape = qml.StronglyEntanglingLayers.shape(
    n_layers=N_LAYERS,
    n_wires=N_QUBITS
)

weights = pnp.array(
    0.05 * np.random.randn(
        *weight_shape
    ),
    requires_grad=True
)


optimizer = qml.AdamOptimizer(
    stepsize=0.08
)


X_vqc_pnp = pnp.array(
    X_train_vqc,
    requires_grad=False
)

y_vqc_pnp = pnp.array(
    y_train,
    requires_grad=False
)


# ---------------------------------------------------------
# TRAIN VQC
# ---------------------------------------------------------

EPOCHS = 20
BATCH_SIZE = 16

print("\nTraining VQC...")


for epoch in range(EPOCHS):

    indices = np.random.permutation(
        len(X_vqc_pnp)
    )

    for start in range(
        0,
        len(indices),
        BATCH_SIZE
    ):

        batch_idx = indices[
            start:start + BATCH_SIZE
        ]

        X_batch = X_vqc_pnp[
            batch_idx
        ]

        y_batch = y_vqc_pnp[
            batch_idx
        ]

        weights, cost = optimizer.step_and_cost(
            lambda w:
                batch_loss(
                    w,
                    X_batch,
                    y_batch
                ),
            weights
        )

    if (epoch + 1) % 5 == 0:

        print(
            f"VQC epoch {epoch + 1}/{EPOCHS} "
            f"- loss: {float(cost):.4f}"
        )


# ---------------------------------------------------------
# 5. TEST VQC
# ---------------------------------------------------------

vqc_predictions = []

for x in X_test_vqc:

    probability = float(
        vqc_probability(
            x,
            weights
        )
    )

    prediction = (
        1
        if probability >= 0.5
        else 0
    )

    vqc_predictions.append(
        prediction
    )


vqc_acc = float(
    np.mean(
        np.array(vqc_predictions)
        == y_test
    )
)


print(
    "\nVQC test accuracy:",
    vqc_acc
)


# Save VQC weights
np.save(
    MODEL_DIR / "vqc_weights.npy",
    np.array(weights)
)


# ---------------------------------------------------------
# 6. SAVE PROJECT INFORMATION
# ---------------------------------------------------------

metadata = {

    "classes": [
        0,
        1
    ],

    "image_size": [
        8,
        8
    ],

    "n_qubits": N_QUBITS,

    "n_layers": N_LAYERS,

    "cnn_test_accuracy":
        float(cnn_acc),

    "vqc_test_accuracy":
        float(vqc_acc),

    "dataset":
        "Scikit-learn Digits dataset (digits 0 and 1)"
}


with open(
    MODEL_DIR / "metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


print("\n--------------------------------")
print("Training completed successfully!")
print("--------------------------------")

print("\nGenerated files:")

for file in MODEL_DIR.iterdir():

    print(
        " -",
        file.name
    )
