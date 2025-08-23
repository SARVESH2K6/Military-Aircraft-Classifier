# app.py (Deep Debugging Version)
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
    input_shape_to_use = (224, 224, 3)
    st.write(f"--- INSIDE build_model: Building model with shape {input_shape_to_use}") # DEBUG PRINT

    data_augmentation = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False, weights='imagenet', input_shape=input_shape_to_use
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
# IMPORTANT: Caching is temporarily disabled for this test by commenting out the decorator
# @st.cache_resource
def load_model_and_classes():
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        
        index_to_label = {str(v): k for k, v in class_indices.items()}
        class_count = len(class_indices)
        
        model = build_model(class_count)
        model.load_weights('final_aircraft_weights.h5')
        
        return model, index_to_label
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None

# Load the resources
model, index_to_label = load_model_and_classes()

# --- Prediction Function ---
def predict(image, model, index_to_label_map):
    st.write("--- INSIDE predict function ---") # DEBUG PRINT
    
    img_resized = image.resize((224, 224))
    st.write(f"1. After PIL resize to (224, 224): Image mode is '{img_resized.mode}', Size is {img_resized.size}") # DEBUG PRINT
    
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    st.write(f"2. After img_to_array: Array shape is {img_array.shape}, Data type is {img_array.dtype}") # DEBUG PRINT
    
    img_array = np.expand_dims(img_array, axis=0)
    st.write(f"3. After expand_dims (adding batch dimension): Shape is {img_array.shape}") # DEBUG PRINT
    
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    st.write(f"4. After preprocess_input: Shape is {img_array.shape}, Min value is {np.min(img_array):.2f}, Max value is {np.max(img_array):.2f}") # DEBUG PRINT
    
    st.write("--- PREDICTION ---") # DEBUG PRINT
    predictions = model.predict(img_array)[0]
    
    top_k_indices = np.argsort(predictions)[-5:][::-1]
    top_k_scores = predictions[top_k_indices]
    
    results = {index_to_label_map[str(i)]: float(s * 100) for i, s in zip(top_k_indices, top_k_scores)}
    
    return results

# --- Streamlit UI ---
st.title("✈️ Military Aircraft Classifier (Debug Mode)")
st.markdown("Upload an image to see the debug output.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if model is not None and uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image to be classified', use_column_width=True)
    st.markdown("---")
    
    # Run the prediction and show debug info
    results = predict(image, model, index_to_label)
    
    # Display the final result
    st.markdown("---")
    st.header("Final Prediction Result")
    top_prediction_class = list(results.keys())[0]
    top_prediction_confidence = list(results.values())[0]
    st.success(f"**Top Prediction:** {top_prediction_class} (Confidence: {top_prediction_confidence:.2f}%)")
    result_df = pd.DataFrame(list(results.items()), columns=['Aircraft', 'Confidence (%)'])
    st.bar_chart(result_df.set_index('Aircraft'))

elif model is None:
    st.warning("Application is offline: The model files could not be loaded.")