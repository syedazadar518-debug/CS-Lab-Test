import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# --- Page Setup ---
st.set_page_config(page_title="AI Face Detector", page_icon="🤖")

st.title("🤖 AI Computer Vision App")
st.write("Upload an image to detect faces using Mediapipe.")

# Sidebar info
st.sidebar.header("System Status")

# Error handling for Mediapipe initialization
try:
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    st.sidebar.success("AI Model Loaded")
except AttributeError:
    st.error("Mediapipe load nahi ho saka. Please check your requirements.txt")

def detect_faces(image):
    img_array = np.array(image.convert('RGB'))
    
    # Initialize detector locally inside function to avoid global attribute errors
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img_array)
        
        annotated_image = img_array.copy()
        count = 0
        
        if results.detections:
            count = len(results.detections)
            for detection in results.detections:
                mp_drawing.draw_detection(annotated_image, detection)
        
        return annotated_image, count

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(input_image, use_container_width=True)
        
    with st.spinner("AI is processing..."):
        try:
            processed_img, face_count = detect_faces(input_image)
            with col2:
                st.subheader("AI Result")
                st.image(processed_img, use_container_width=True)
            
            if face_count > 0:
                st.success(f"Detected {face_count} faces!")
            else:
                st.info("No faces found.")
        except Exception as e:
            st.error(f"Processing Error: {e}")

else:
    st.info("Please upload an image to start detection.")

st.markdown("---")
st.caption("No API Key Required | Running on Local Pre-trained Model")
