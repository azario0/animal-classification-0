import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import json

# Load the model
model = load_model('my_model.h5')

# Load class names from file
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

def preprocess_image(img):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # Resize the image to match the input size of the model
    img_resized = cv2.resize(img_gray, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Apply Canny edge detection
    edges = cv2.Canny(img_resized, 100, 200)
    
    # Normalize the image
    edges = edges / 255.0
    
    # Convert to uint8 to avoid errors
    edges = (edges * 255).astype(np.uint8)
    
    # Convert edges to 3 channels
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Expand dimensions to match the input shape of the model
    edges_3ch = np.expand_dims(edges_3ch, axis=0)
    
    return edges_3ch, edges

def predict_image(model, img):
    # Preprocess the image
    processed_image, edges = preprocess_image(img)
    
    # Make a prediction
    prediction = model.predict(processed_image)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)
    
    return predicted_class, edges

st.title('Animal Classification App')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    
    # Display the original image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict the class of the image
    predicted_class, edges = predict_image(model, image)
    
    # Display the processed image with edges
    st.image(edges, caption='Processed Image with Edges', use_column_width=True)
    
    # Display the predicted class
    st.write(f'Predicted class: {class_names[predicted_class[0]]}')
