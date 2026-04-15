import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# --- Page Config ---
st.set_page_config(page_title="AI Vision Lab", layout="wide")

st.title("🎨 AI Pixel Vision Lab")
st.write("Explore how Computer Vision 'sees' images through different processing filters.")

# --- Sidebar ---
st.sidebar.header("Filter Settings")
mode = st.sidebar.selectbox(
    "Select Vision Mode", 
    ["Original", "Grayscale", "Edge Detection", "Pencil Sketch", "Heatmap (Fake)", "Blur Vision"]
)

# --- Logic Functions ---
def apply_filter(img, mode):
    if mode == "Grayscale":
        return ImageOps.grayscale(img)
    
    elif mode == "Edge Detection":
        # Computer Vision ka basic principle: Finding boundaries
        return img.convert("L").filter(ImageFilter.FIND_EDGES)
    
    elif mode == "Pencil Sketch":
        # Gray -> Blur -> Invert -> Dodge
        gray = ImageOps.grayscale(img)
        inverted = ImageOps.invert(gray)
        blurred = inverted.filter(ImageFilter.GaussianBlur(radius=5))
        final = Image.blend(gray, blurred, alpha=0.5)
        return final

    elif mode == "Blur Vision":
        return img.filter(ImageFilter.GaussianBlur(radius=10))

    elif mode == "Heatmap (Fake)":
        # Convert to grayscale then apply a 'Hot' look
        gray = ImageOps.grayscale(img)
        return ImageOps.colorize(gray, black="blue", white="red")

    return img

# --- UI Layout ---
uploaded_file = st.file_uploader("Upload an image to transform...", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Load Image
    img = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Source Image")
        st.image(img, use_container_width=True)
        
    with st.spinner(f"Applying {mode}..."):
        result = apply_filter(img, mode)
        
    with col2:
        st.subheader(f"{mode} Result")
        st.image(result, use_container_width=True)
        
    # Download Button
    st.sidebar.markdown("---")
    st.sidebar.download_button("Download Result", data=uploaded_file, file_name="transformed.png")

else:
    st.info("Please upload an image to see the AI Vision magic.")

# --- Footer Info for CS Students ---
with st.expander("How this works (CS Logic)"):
    st.write("""
        * **Grayscale:** Luma transformation using `Y = 0.299R + 0.587G + 0.114B`.
        * **Edge Detection:** Sobel/Prewitt operator logic filters out low-frequency pixels.
        * **Pencil Sketch:** Uses Gaussian Blur and Color Blending to simulate hand-drawn art.
    """)

st.markdown("---")
st.caption("Pure Python | No Heavy Models | 100% Stability")
