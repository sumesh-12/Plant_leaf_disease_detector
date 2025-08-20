import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import base64
import cv2  # For image processing visualizations

# --- Page Configuration ---
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

# --- Function to set blurred background image ---
def set_jpg_as_page_bg(jpg_file):
    """Sets a blurred background image with frosted-glass effect containers."""
    try:
        with open(jpg_file, "rb") as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        page_bg_img = f"""
        <style>
        /* Background blur layer */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url("data:image/jpeg;base64,{bin_str}") no-repeat center center fixed;
            background-size: cover;
            filter: blur(12px);
            z-index: -2;
        }}

        /* Dark overlay for readability */
        .stApp::after {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.35);
            z-index: -1;
        }}

        /* Universal frosted-glass effect */
        .stApp,
        .stAppViewContainer,
        .main,
        .block-container {{
            background: rgba(255, 255, 255, 0.12) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
        }}

        /* Sidebar frosted-glass */
        section[data-testid="stSidebar"] {{
            background: rgba(255, 255, 255, 0.12) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
        }}

        /* Transparent header */
        .stApp > header {{
            background: transparent !important;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Background image '{jpg_file}' not found.")

# --- Call the function to set the background ---
set_jpg_as_page_bg('background.jpg')

# --- Model Loading ---
@st.cache_resource
def load_plant_model():
    """Loads the pre-trained plant disease detection model."""
    model_path = 'plant_disease_model_densenet.h5'
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

model = load_plant_model()

# --- Class Names ---
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

# --- Disease Information and Suggestions ---
disease_info = {
    "healthy": "The plant appears to be healthy. Continue with regular care and monitoring.",
    "angular_leaf_spot": "A fungal disease causing angular spots on leaves. Suggestions: Improve air circulation, avoid overhead watering, and consider using a fungicide.",
    "anthracnose": "A fungal disease causing dark, sunken lesions. Suggestions: Prune affected areas, destroy infected plant debris, and apply a fungicide.",
    "bean_rust": "A fungal disease causing rust-colored pustules. Suggestions: Ensure proper spacing for air circulation, avoid wet foliage, and use resistant varieties or fungicides.",
    "mosaic_virus": "A viral disease causing mottled yellow and green patterns. Suggestions: There is no cure. Remove and destroy infected plants to prevent spread. Control aphids, which transmit the virus.",
    "blight": "A common fungal or bacterial disease causing rapid browning and death of plant tissue. Suggestions: Improve air circulation, apply appropriate fungicides or bactericides, and remove infected parts.",
    "common_rust": "A fungal disease common in maize, causing small, cinnamon-brown pustules. Suggestions: Plant resistant hybrids and apply fungicides if necessary.",
    "downy_mildew": "A disease caused by an oomycete, leading to yellow spots and white mold on the underside of leaves. Suggestions: Reduce humidity, improve air circulation, and apply a targeted fungicide.",
    "gray_leaf_spot": "A fungal disease causing small, rectangular gray lesions on maize leaves. Suggestions: Use resistant hybrids and practice crop rotation.",
    "lethal_necrosis": "A severe viral disease in maize. Suggestions: There is no cure. Control the insect vectors that spread the virus and remove infected plants immediately.",
    "my_streak_virus": "A viral disease in maize transmitted by leafhoppers. Suggestions: Control leafhopper populations and use resistant varieties.",
    "bacterial_spot": "A bacterial disease causing small, water-soaked spots on tomato leaves. Suggestions: Avoid overhead watering, use disease-free seeds, and apply copper-based bactericides.",
    "early_blight": "A fungal disease causing 'target spot' lesions on tomato leaves. Suggestions: Mulch around plants, prune lower leaves, and apply fungicides.",
    "late_blight": "A devastating oomycete disease affecting tomatoes, causing large, dark lesions. Suggestions: Ensure good air circulation, avoid overhead watering, and apply preventative fungicides.",
    "leaf_mold": "A fungal disease causing yellow spots on the upper leaf surface and olive-green mold underneath. Suggestions: Improve ventilation and reduce humidity.",
    "septoria_leaf_spot": "A fungal disease causing small, circular spots with dark borders. Suggestions: Remove infected leaves, mulch, and apply fungicides.",
    "spider_mite": "A common pest, not a disease, that causes stippling on leaves. Suggestions: Use insecticidal soap or miticides. Increase humidity as mites thrive in dry conditions.",
    "target_spot": "A fungal disease causing target-like spots on tomato leaves. Suggestions: Improve air circulation and apply fungicides.",
    "yellow_leaf_curl_virus": "A viral disease transmitted by whiteflies, causing yellowing and curling of leaves. Suggestions: Control whitefly populations and remove infected plants."
}

# --- Image Preprocessing ---
def preprocess_image(image):
    """Preprocesses the uploaded image to be compatible with the model."""
    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Main Page UI ---
st.title("🌿 Plant Leaf Disease Detector")

st.subheader("Why to treat plant diseases?")
st.info(
    """
    Plant leaf diseases need to be eradicated to prevent significant economic losses, reduced crop yields, and potential food shortages. Early detection and treatment are crucial to minimize the impact of diseases on plant health and productivity.
    """
)

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
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    with col2:
        if st.button('Diagnose Leaf'):
            if model is not None:
                with st.spinner('Analyzing the leaf...'):
                    try:
                        processed_image = preprocess_image(image)
                        prediction = model.predict(processed_image)
                        
                        top_indices = np.argsort(prediction[0])[-3:][::-1]
                        top_classes = [class_names[i] for i in top_indices]
                        top_confidences = [prediction[0][i] for i in top_indices]

                        predicted_class_name = top_classes[0]
                        confidence = top_confidences[0]
                        
                        plant, disease = predicted_class_name.split('___')
                        plant = plant.replace('_', ' ').title()
                        disease_key = disease.lower()
                        disease_display = disease.replace('_', ' ').title()

                        st.success(f"**Diagnosis Complete!**")
                        st.metric("Top Prediction", f"{plant} - {disease_display}", f"Confidence: {confidence:.2%}")

                        if disease_key in disease_info:
                            st.info(f"**Suggestion:** {disease_info[disease_key]}")
                        else:
                            st.warning("No specific suggestion available for this condition.")

                        st.subheader("Top Predictions")
                        df = pd.DataFrame({
                            'Disease': [c.split('___')[1].replace('_', ' ').title() for c in top_classes],
                            'Confidence': top_confidences
                        })
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index('Disease'))

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
            else:
                st.warning("The model is not loaded.")
    
    # --- Visualization Section ---
    st.markdown("---")
    with st.expander("🔬 Visualizing the Image Processing Pipeline"):
        st.write("Here's how we process your image before feeding it to the AI model.")
        
        opencv_image = np.array(image.convert('RGB'))
        opencv_image = opencv_image[:, :, ::-1].copy()

        viz_col1, viz_col2, viz_col3 = st.columns(3)

        with viz_col1:
            st.image(image, caption="1. Original Image", use_column_width=True)

        with viz_col2:
            resized_img = cv2.resize(opencv_image, (128, 128))
            st.image(resized_img, channels="BGR", caption="2. Resized to 128x128", use_column_width=True)

        with viz_col3:
            normalized_img = resized_img / 255.0
            st.image(normalized_img, channels="BGR", caption="3. Normalized Pixels", use_column_width=True)

        st.write("---")
        st.write("Below are examples of **Data Augmentation** we use during training.")

        aug_col1, aug_col2, aug_col3 = st.columns(3)

        with aug_col1:
            center = (resized_img.shape[1]//2, resized_img.shape[0]//2)
            angle = 30
            scale = 1.0
            rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
            rotated_img = cv2.warpAffine(resized_img, rot_mat, (resized_img.shape[1], resized_img.shape[0]))
            st.image(rotated_img, channels="BGR", caption="Example: Rotated", use_column_width=True)

        with aug_col2:
            flipped_img = cv2.flip(resized_img, 1)
            st.image(flipped_img, channels="BGR", caption="Example: Flipped", use_column_width=True)
        
        with aug_col3:
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            st.image(gray_img, caption="Example: Grayscale", use_column_width=True)

st.markdown("---")

# --- Techniques used in model section ---
st.subheader("The techniques used in model:")
st.markdown(
    """
    - **Transfer Learning with DenseNet121**: Pre-trained model with strong feature extraction.
    - **Custom Classifier Architecture**: New layers added to classify plant diseases.
    - **Two-Phase Training Strategy**: First trained new layers, then fine-tuned whole model.
    - **Smart Optimization and Monitoring**: Callbacks for best model saving, early stopping, and LR scheduling.
    """
)

st.markdown("---")

# --- SDG 3 section ---
col1_sdg, col2_sdg = st.columns([1, 2])

with col1_sdg:
    try:
        st.image("Sustainable_Development_Goal_03GoodHealth.jpg")
    except FileNotFoundError:
        st.warning("SDG image not found. Make sure 'Sustainable_Development_Goal_03GoodHealth.jpg' is in the same folder.")

with col2_sdg:
    st.subheader("Contribution to SDG 3: Good Health and Well-Being")
    st.write(
        """
        By improving food safety and security, this project directly contributes to SDG 3: Good Health and Well-Being.
        Identifying plant diseases early enables targeted treatment, protects crop yields, and reduces excessive pesticide use.
        This results in safer food and supports global community welfare.
        """
    )
