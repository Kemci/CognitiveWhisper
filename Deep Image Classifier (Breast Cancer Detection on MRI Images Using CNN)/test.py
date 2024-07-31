import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('./models/malignant_benign_model.h5')

# Preprocessing function (adjust based on your model requirements)
def preprocess_image(image):
    image = image.resize((224, 224))  # Example size
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to classify image
def classify_image(image_path):
    image = Image.open(image_path)
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    class_index = np.argmax(prediction)
    return class_index

# Function to load and display image
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((250, 250))
        img = ImageTk.PhotoImage(image)
        img_label.config(image=img)
        img_label.image = img

        # Perform classification
        class_index = classify_image(file_path)
        result_label.config(text=f'Predicted Class: {class_index}')

# Set up Tkinter window
root = tk.Tk()
root.title("Image Classification")

# Create and place widgets
load_btn = tk.Button(root, text="Load Image", command=load_image)
load_btn.pack()

img_label = Label(root)
img_label.pack()

result_label = Label(root, text="Predicted Class: ")
result_label.pack()

# Run the Tkinter event loop
root.mainloop()
