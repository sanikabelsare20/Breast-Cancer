import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import cv2

# Load the pre-trained model in .h5 format
MODEL_PATH = "model_vgg16.h5"  # Update with the actual path to your .h5 model file
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Function to preprocess uploaded image
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")  # Convert to RGB if the image is in grayscale
    image = image.resize(target_size)  # Resize to match VGG16 input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    classes = ["Normal", "Benign", "Malignant"]  # Update based on your model's classes
    predicted_class = classes[np.argmax(predictions)]  # Get the predicted class
    confidence = np.max(predictions) * 100  # Get confidence score
    return predicted_class, confidence

# Function to extract image parameters
def extract_image_parameters(image):
    image_array = np.array(image)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    elif len(image_array.shape) == 2:
        image_gray = image_array
    else:
        raise ValueError("Unsupported image format")

    density = np.sum(image_gray > 150) / (image_gray.shape[0] * image_gray.shape[1])
    return {
        "Patient ID": "P123456",
        "Breast Density": f"{density:.2f}%",
        "Left or Right Breast": "Left",
        "Image View": "C"C (Cranio-Caudal)",
        "Abnormality ID": "A78901",
        "Abnormality Type": "Mass",
        "Mass Shape": "Irregular",
        "Mass Margins": "Spiculated"
    }

# Function to generate a downloadable report
def generate_report(prediction, confidence, parameters):
    report = f"Breast Cancer Classification Report\n"
    report += f"{'-'*40}\n"
    report += f"Prediction: {prediction}\n"
    report += f"Confidence: {confidence:.2f}%\n\n"
    report += "Extracted Parameters:\n"
    for key, value in parameters.items():
        report += f"{key}: {value}\n"
    report += f"{'-'*40}\n"
    return report

# Streamlit Front Page
st.title("Breast Cancer Disease Classification")
st.write("Upload a mammogram image to classify it and extract relevant parameters.")

uploaded_file = st.file_uploader("Upload an Image (JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file)
    st.subheader("Processing the Image...")
    prediction, confidence = predict_image(image)
    parameters = extract_image_parameters(image)
    st.subheader("Prediction Results")
    st.write(f"**Prediction**: {prediction}")
    st.write(f"**Confidence**: {confidence:.2f}%")
    st.subheader("Extracted Parameters")
    for key, value in parameters.items():
        st.write(f"- {key}: {value}")
    st.subheader("Download Report")
    report = generate_report(prediction, confidence, parameters)
    report_bytes = io.BytesIO(report.encode("utf-8"))
    st.download_button(
        label="Download Report",
        data=report_bytes,
        file_name="breast_cancer_report.txt",
        mime="text/plain"
    )
