import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="Pro Face Detector", layout="wide")

st.title("🎯 Multi-Face AI Detector")

# --- Sidebar for Tuning ---
st.sidebar.header("Model Tuning")
# Scale factor ko 1.05 ya 1.1 rakhein taakay chote faces bhi milain
sf = st.sidebar.slider("Sensitivity (Scale Factor)", 1.01, 1.50, 1.1, 0.01)
mn = st.sidebar.slider("Min Neighbors", 1, 10, 4)

def detect_faces(image):
    # Image ko process karne ke liye format set karein
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Pre-trained model load karein
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detection logic: scaleFactor ko adjust karne se multiple faces milte hain
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=sf, 
        minNeighbors=mn, 
        minSize=(30, 30)
    )
    
    annotated_image = img_array.copy()
    for (x, y, w, h) in faces:
        # Draw box
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 4)
        
    return annotated_image, len(faces)

# --- UI Layout ---
tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Live Camera"])

with tab1:
    uploaded_file = st.file_uploader("Upload Image Here", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        img = Image.open(uploaded_file)
        # Force processing on upload
        res_img, count = detect_faces(img)
        
        c1, c2 = st.columns(2)
        c1.image(img, caption="Original", use_container_width=True)
        c2.image(res_img, caption=f"Detected: {count} faces", use_container_width=True)
        
        if count > 0:
            st.success(f"AI found {count} faces!")

with tab2:
    cam_file = st.camera_input("Take a photo")
    if cam_file:
        img_cam = Image.open(cam_file)
        res_cam, count_cam = detect_faces(img_cam)
        st.image(res_cam, caption=f"Detected: {count_cam} faces")

st.markdown("---")
st.caption("Tip: Agar upload mein faces detect nahi ho rahay, to Sidebar se 'Sensitivity' ko 1.05 ya 1.1 par set karein.")
