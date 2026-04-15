import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Page Setup ---
st.set_page_config(page_title="AI Face Detector", page_icon="📸")

st.title("📸 Ultra-Stable AI Face Detector")
st.write("Yeh app OpenCV ka built-in model use karti hai jo Streamlit par 100% stable chalta hai.")

def detect_faces(image):
    # Convert PIL to OpenCV format
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Load the pre-trained Haar Cascade model (Built-in in OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles
    annotated_image = img_array.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (255, 0, 0), 5)
        
    return annotated_image, len(faces)

# --- UI ---
uploaded_file = st.file_uploader("Koi bhi tasveer upload karein...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)
        
    with st.spinner("AI Detection jari hai..."):
        processed_img, face_count = detect_faces(input_image)
        
    with col2:
        st.subheader("AI Detection")
        st.image(processed_img, use_container_width=True)
        
    if face_count > 0:
        st.success(f"Mubarak ho! AI ne {face_count} chehray pehchan liye hain.")
    else:
        st.warning("Koi chehra nahi mila. Dubara koshish karein.")

else:
    st.info("Sidebar ya screen se tasveer select karein.")

st.markdown("---")
st.caption("No API | No Mediapipe Errors | 100% Free OpenCV Model")
