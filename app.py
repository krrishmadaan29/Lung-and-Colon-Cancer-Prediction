import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
import cv2
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Lung & Colon Cancer Classifier",
    page_icon=":microscope:",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .title {
        color: #2a9d8f;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    .upload-box {
        border: 2px dashed #264653;
        border-radius: 10px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
        background-color: #f8f9fa;
    }
    .prediction-box {
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        background-color: #e9f5f4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 8px 0;
        background: linear-gradient(90deg, #e9c46a, #f4a261);
    }
    .stButton>button {
        background-color: #2a9d8f;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Class names
CLASSES = [
    'Colon Adenocarcinoma',
    'Benign Colon Tissue',
    'Lung Adenocarcinoma', 
    'Benign Lung Tissue',
    'Lung Squamous Cell Carcinoma'
]

# Image preprocessing functions
def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced

def preprocess_image(image):
    image = cv2.resize(image, (227, 227))
    image = enhance_image(image)
    image = image / 255.0
    image = (image - np.mean(image)) / np.std(image)
    return image

# Load model with caching
@st.cache_resource
def load_app_model():
    try:
        model = keras_load_model('alexnet_lung_colon_final.h5')
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.error("Please make sure 'alexnet_lung_colon_final.h5' is in the correct directory.")
        return None

def main():
    st.markdown('<h1 class="title">Cancer Histopathology Classifier</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; color: #6c757d;'>
        Upload a histopathology image to classify it into one of 5 cancer/tissue types
    </p>
    """, unsafe_allow_html=True)

    # File uploader section
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag & drop an image or click to browse",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    model = load_app_model()
    
    if uploaded_file is not None and model is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert to numpy array and preprocess
        img_array = np.array(image)
        processed_img = preprocess_image(img_array)
        input_img = np.expand_dims(processed_img, axis=0)
        
        # Make prediction
        if st.button('Classify Image', type='primary'):
            with st.spinner('Analyzing...'):
                try:
                    predictions = model.predict(input_img)
                    predicted_class = np.argmax(predictions)
                    confidence = np.max(predictions) * 100
                
                    # Display results
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.subheader("Results")
                    st.success(f"**Prediction**: {CLASSES[predicted_class]}")
                    st.success(f"**Confidence**: {confidence:.2f}%")
                    
                    st.subheader("Class Probabilities")
                    for i, (cls, prob) in enumerate(zip(CLASSES, predictions[0])):
                        st.write(f"{cls}:")
                        st.markdown(f"""
                        <div class="confidence-bar" style="width: {prob*100:.1f}%"></div>
                        <span>{prob*100:.2f}%</span>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()