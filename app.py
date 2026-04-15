import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="AI Vision App", layout="centered")

# --- Mediapipe Face Detection Setup ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def process_image(image):
    # Convert PIL image to OpenCV format
    img_array = np.array(image.convert('RGB'))
    
    # Initialize Face Detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        # Process the image
        results = face_detection.process(img_array)

        # Draw detections on the image
        annotated_image = img_array.copy()
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(annotated_image, detection)
            return annotated_image, len(results.detections)
        else:
            return annotated_image, 0

# --- User Interface (UI) ---
st.title("🤖 AI Computer Vision App")
st.write("Upload an image to detect faces automatically using a pre-trained model.")

# Sidebar for Instructions
st.sidebar.header("About This App")
st.sidebar.info("Ye app **Mediapipe** use karti hai jo k aik free, local AI model hai. Is mein koi API key nahi chahiye.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Process and show results
    with st.spinner('AI model processing...'):
        result_img, count = process_image(image)
        
    with col2:
        st.subheader("AI Detection")
        st.image(result_img, use_container_width=True)
        
    # Show Success Message
    if count > 0:
        st.success(f"Detections Successful! Total {count} faces found.")
    else:
        st.warning("No faces detected in this image.")

else:
    st.info("Please upload an image to start.")

# Footer
st.markdown("---")
st.caption("Powered by Streamlit & Mediapipe | No API Required")
