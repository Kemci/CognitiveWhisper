import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load your model
model = tf.keras.models.load_model('./models/malignant_benign_model.h5')

def preprocess_image(image):
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    # Resize the image to the required size
    image_resized = cv2.resize(image_gray, (256, 256))
    # Expand dimensions to fit the model input
    image_expanded = np.expand_dims(image_resized, axis=(0, -1))
    return image_expanded / 255.0

def main():
    # Set the page configuration
    st.set_page_config(
        page_title="Breast Cancer Detection",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #1c1c1c;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        color: #e0e0e0;
    }
    .title {
        font-size: 2.5rem;
        color: #ff6f61;
    }
    .description {
        font-size: 1.2rem;
        color: #d3d3d3;
    }
    .prediction {
        font-size: 1.5rem;
        color: #4caf50;
    }
    .uploaded-image {
        max-width: 50%;
        margin: 20px 0;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .sidebar .sidebar-content {
        background-color: #2e2e2e;
        color: #e0e0e0;
        transition: all 0.3s ease;
    }
    .sidebar .sidebar-content:hover {
        background-color: #3a3a3a;
    }
    .sidebar .sidebar-content h2 {
        font-size: 2rem;
        color: #ff6f61;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Add a sidebar with useful sections
    st.sidebar.markdown('<div class="sidebar-content"><h2>Ikemka\'s MRI Model</h2></div>', unsafe_allow_html=True)
    st.sidebar.header("About")
    st.sidebar.write("""
    This application uses a Convolutional Neural Network (CNN) to predict whether an MRI image of breast tissue is malignant or benign. 
    Upload an image to get started.
    """)
    
    st.sidebar.header("Contact")
    st.sidebar.write("""
    For more information or support, please contact:
    - Email: kemciike@gmail.com
    - Phone: +234 817 012 5350
    """)
    
    st.sidebar.header("Useful Links")
    st.sidebar.write("[Breast Cancer Research Foundation](https://www.bcrf.org/)")
    st.sidebar.write("[American Cancer Society](https://www.cancer.org/)")
    st.sidebar.write("[World Health Organization](https://www.who.int/)")

    # Main title and description
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="title">Breast Cancer Detection on MRI Images using CNN</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Upload an MRI image to predict whether it is malignant or benign.</div>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "bmp"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=False, width=600)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        yhat = model.predict(processed_image)
        prediction = 'Malignant' if yhat > 0.5 else 'Benign'
        
        # Display prediction
        st.markdown(f'<div class="prediction">Prediction: {prediction}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
