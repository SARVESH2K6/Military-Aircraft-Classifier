# app.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import streamlit as st
import tensorflow as tf
import json
import numpy as np
import pandas as pd
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Military Aircraft Classifier",
    page_icon="✈️",
    layout="wide"
)

# --- Function to Build the Model Architecture ---
def build_model(class_count):
    """
    Builds the exact same model architecture that was used for training.
    """
    # --- DEBUGGING STEP ---
    # This will print to the Streamlit deploy log to confirm the shape.
    input_shape_to_use = (224, 224, 3)
    print(f"DEBUG LOG: Building model with input shape: {input_shape_to_use}")
    # --------------------

    data_augmentation = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape_to_use
    )
    base_model.trainable = True

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape_to_use),
        data_augmentation,
        base_model,
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.Dense(class_count, activation='softmax')
    ])
    return model

# --- Model and Class Loading Function ---
@st.cache_resource
def load_model_and_classes():
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        
        index_to_label = {str(v): k for k, v in class_indices.items()}
        class_count = len(class_indices)
        
        model = build_model(class_count)
        # Ensure this is the correct filename
        model.load_weights('model.weights.h5')
        
        return model, index_to_label
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None

# Load the resources
model, index_to_label = load_model_and_classes()

# --- Prediction Function ---
def predict(image, model, index_to_label_map, top_k=5):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    predictions = model.predict(img_array)[0]
    
    top_k_indices = np.argsort(predictions)[-top_k:][::-1]
    top_k_scores = predictions[top_k_indices]
    
    results = {index_to_label_map[str(i)]: float(s * 100) for i, s in zip(top_k_indices, top_k_scores)}
    
    return results

# --- Streamlit App Layout ---

# Sidebar
st.sidebar.title("About & Controls")
st.sidebar.info(
    "This is a deep learning application for classifying military aircraft, "
    "built as a portfolio project to demonstrate a complete machine learning workflow.\n\n"
    "**Model:** Fine-tuned `EfficientNetB3`\n\n"
    "**Final Test Accuracy:** 91.62%\n\n"
    "**Number of Classes:** 40\n\n"
    "For a detailed write-up and the complete source code, please visit the GitHub repository:"
)
st.sidebar.markdown("[GitHub Repository](https://github.com/SARVESH2K6/Military-Aircraft-Classifier)")

# Confidence Threshold Slider
st.sidebar.header("Prediction Controls")
confidence_threshold = st.sidebar.slider(
    "Set Confidence Threshold (%)", 
    min_value=0, 
    max_value=100, 
    value=50, # Default value
    step=5
)

# Main Page
# --- DEBUGGING STEP ---
# Added a version number to the title
st.title("✈️ Military Aircraft Classifier v1.1")
# --------------------
st.markdown("Upload an image of a military aircraft, and the model will predict its class.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if model is not None and uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption='Image to be classified', use_column_width=True)
    
    st.markdown("---")

    with st.spinner('Analyzing the aircraft...'):
        results = predict(image, model, index_to_label)
    
    top_prediction_class = list(results.keys())[0]
    top_prediction_confidence = list(results.values())[0]

    if top_prediction_confidence >= confidence_threshold:
        st.success(f"**Top Prediction:** {top_prediction_class}")
        st.info(f"**Confidence:** {top_prediction_confidence:.2f}%")
    else:
        st.warning(f"**Low Confidence Prediction:** The model is not confident about its top guess.")
        st.info(f"**Best Guess:** {top_prediction_class} (Confidence: {top_prediction_confidence:.2f}%)")
        st.markdown("The chart below shows the model's uncertainty among the top predictions.")
    
    st.markdown("### Top Predictions")
    result_df = pd.DataFrame(list(results.items()), columns=['Aircraft', 'Confidence (%)'])
    st.bar_chart(result_df.set_index('Aircraft'))

elif model is None:
    st.warning("Application is offline: The model files could not be loaded.")