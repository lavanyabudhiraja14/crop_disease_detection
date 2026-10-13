"""
Prediction utilities for the Streamlit application.
"""

import numpy as np
import tensorflow as tf
from PIL import Image


def load_trained_model(model_path):
    """Load a saved Keras model (.h5 or SavedModel directory)."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model at '{model_path}': {e}") from e


def preprocess_image(image_file, target_size=(224, 224)):
    """
    Read an uploaded image file and return a float32 array shaped (1, H, W, 3).

    The stream position is reset to the beginning so the same file object
    can be reused (e.g. after st.image() has already consumed it).

    Pixel values are kept in [0, 255] because the model includes a
    built-in preprocess_input layer that handles normalisation.
    """
    # Reset stream in case it was already consumed (e.g. by st.image)
    if hasattr(image_file, 'seek'):
        image_file.seek(0)

    try:
        img = Image.open(image_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size, Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32)   # [0, 255]
        img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)
        return img_array
    except Exception as e:
        raise ValueError(f"Could not process the uploaded image: {e}") from e


def predict_image(model, img_array, classes):
    """
    Run inference and return a list of dicts sorted by confidence (descending).

    Each dict has keys 'class' and 'confidence'.
    """
    predictions = model.predict(img_array, verbose=0)
    pred_probs = predictions[0]

    sorted_indices = np.argsort(pred_probs)[::-1]

    results = []
    for idx in sorted_indices:
        results.append({
            "class": classes[idx],
            "confidence": float(pred_probs[idx]),
        })
    return results
