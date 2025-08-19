import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# --- Model Loading ---
# Use st.cache_resource to load the model only once, preventing reloads on every interaction.
@st.cache_resource
def load_plant_model():
    """Loads the pre-trained plant disease detection model."""
    # IMPORTANT: Make sure the model filename matches the one you saved.
    model_path = 'plant_disease_model_densenet.h5' 
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

model = load_plant_model()

# --- Class Names ---
# IMPORTANT: This list MUST be in the same order as your training data folders.
# You can get this order from the output of your Phase 1 preprocessing script.
class_names = [
    'beans___angular_leaf_spot', 'beans___anthracnose', 'beans___bean_rust', 
    'beans___healthy', 'beans___mosaic_virus', 'maize___blight', 
    'maize___common_rust', 'maize___downy_mildew', 'maize___gray_leaf_spot', 
    'maize___healthy', 'maize___lethal_necrosis', 'maize___my_streak_virus', 
    'tomato___bacterial_spot', 'tomato___early_blight', 'tomato___healthy', 
    'tomato___late_blight', 'tomato___leaf_mold', 'tomato___mosaic_virus', 
    'tomato___septoria_leaf_spot', 'tomato___spider_mite', 'tomato___target_spot', 
    'tomato___yellow_leaf_curl_virus'
]
# This is a generic list from the PlantVillage dataset. You may need to update it.

# --- Image Preprocessing ---
def preprocess_image(image):
    """Preprocesses the uploaded image to be compatible with the model."""
    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = img_array / 255.0 # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# --- UI Elements ---
st.title("🌿 Plant Leaf Disease Detector")
st.write(
    "Upload an image of a plant leaf to identify its health status. "
    "Our advanced model will analyze the image and predict the disease."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    if st.button('Diagnose Leaf'):
        if model is not None:
            with st.spinner('Analyzing the leaf... This may take a moment.'):
                try:
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image)
                    predicted_class_index = np.argmax(prediction)
                    predicted_class_name = class_names[predicted_class_index]

                    plant, disease = predicted_class_name.split('___')
                    plant = plant.replace('_', ' ').title()
                    disease = disease.replace('_', ' ').title()

                    st.success(f"**Diagnosis Complete!**")
                    st.markdown(f"**Plant Type:** `{plant}`")
                    st.markdown(f"**Predicted Condition:** `{disease}`")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("The model is not loaded. Please ensure the model file is in the correct directory.")

