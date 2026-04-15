import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# --- Page Setup ---
st.set_page_config(page_title="AI Vision Detector", page_icon="🤖")

# Initialize Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_faces(image):
    # Convert PIL to OpenCV format (RGB to BGR)
    img_array = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Run Detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # Draw Results
        annotated_image = img_array.copy()
        count = 0
        if results.detections:
            count = len(results.detections)
            for detection in results.detections:
                mp_drawing.draw_detection(annotated_image, detection)
                
        return annotated_image, count

# --- UI Design ---
st.title("🤖 Free AI Computer Vision App")
st.markdown("""
    This app uses a **pre-trained AI model** to detect faces in images.
    * **No API Key** required.
    * **No Payment** required.
    * **100% Privacy** (processed on the fly).
""")

st.sidebar.header("Settings")
st.sidebar.write("Using: Mediapipe Face Detection")

uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Open and process image
    input_image = Image.open(uploaded_file)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(input_image, use_container_width=True)
        
    with st.spinner("AI is thinking..."):
        # Process detection
        processed_img, face_count = detect_faces(input_image)
        
    with col2:
        st.subheader("AI Result")
        st.image(processed_img, use_container_width=True)
        
    if face_count > 0:
        st.success(f"Successfully detected {face_count} face(s)!")
    else:
        st.info("No faces detected in this image.")

else:
    st.warning("👈 Please upload an image from the sidebar or main screen.")

st.markdown("---")
st.caption("Developed for Streamlit Cloud Deployment | Local Inference Only")
