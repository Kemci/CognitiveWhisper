import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load your model
model = tf.keras.models.load_model('./models/happy&sadmodel.h5')


def preprocess_image(image):
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    # Resize the image to the required size
    image_resized = cv2.resize(image_gray, (256, 256))
    # Expand dimensions to fit the model input
    image_expanded = np.expand_dims(image_resized, axis=(0, -1))
    return image_expanded / 255.0

def main():
    st.title("Image Classification with TensorFlow")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        yhat = model.predict(processed_image)
        prediction = 'Malignant' if yhat > 0.5 else 'Benign'
        
       # st.write(f"Prediction: {prediction}")
        st.markdown(f"### Prediction: {prediction}")


if __name__ == "__main__":
    main()
