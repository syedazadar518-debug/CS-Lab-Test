import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Pro AI Vision", 
    page_icon="🎯", 
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.title("🎯 Pro AI Face Analytics")
st.write("Detect faces with precision using OpenCV's pre-trained Haar Cascade model.")

# --- Sidebar Controls ---
st.sidebar.header("🛠️ Model Settings")
st.sidebar.info("In settings se aap detection ki accuracy barha sakte hain.")

# Tuning parameters for the model
scale_factor = st.sidebar.slider("Scale Factor (Sensitivity)", 1.1, 1.5, 1.2, 0.1)
min_neighbors = st.sidebar.slider("Min Neighbors (Quality)", 3, 10, 5)
box_color = st.sidebar.color_picker("Detection Box Color", "#00FF00")

# Convert hex color to BGR for OpenCV
hex_color = box_color.lstrip('#')
bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def detect_faces(image):
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Advanced: Histogram Equalization for better detection in low light
    gray = cv2.equalizeHist(gray)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=scale_factor, 
        minNeighbors=min_neighbors, 
        minSize=(30, 30)
    )
    
    annotated_image = img_array.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), bgr_color, 4)
        # Adding a small label
        cv2.putText(annotated_image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr_color, 2)
        
    return annotated_image, len(faces)

# --- Input Methods ---
tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Live Camera"])

with tab1:
    uploaded_file = st.file_uploader("Choose a photo...", type=['jpg', 'jpeg', 'png'], key="upload")

with tab2:
    camera_file = st.camera_input("Take a picture")

# Combined logic for both inputs
source = uploaded_file if uploaded_file else camera_file

if source is not None:
    input_image = Image.open(source)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Source")
        st.image(input_image, use_container_width=True)
        
    with st.spinner("Analyzing pixels..."):
        processed_img, face_count = detect_faces(input_image)
        
    with col2:
        st.subheader("AI Detection Result")
        st.image(processed_img, use_container_width=True)
        
    # --- Metrics ---
    st.markdown("---")
    m1, m2 = st.columns(2)
    m1.metric("Faces Detected", face_count)
    m2.metric("Model Status", "Active (Local)")

    if face_count > 0:
        st.balloons()
        st.success(f"Successfully identified {face_count} faces in the image.")
    else:
        st.warning("No faces detected. Try adjusting the 'Scale Factor' in the sidebar.")

else:
    st.info("Start by uploading a photo or using your webcam.")

# --- Footer ---
st.markdown("---")
st.caption("Computer Science Project | OpenCV | Streamlit Deployment Ready")
