import streamlit as st
import cv2
import numpy as np
from PIL import Image
import processors  # Import the logic file

st.set_page_config(page_title="Group Cartoonizer", layout="wide")

st.title("🎨 Classical vs. AI Cartoonizer")
st.markdown("### User Interface & Integration Module")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
processing_mode = st.sidebar.radio("Select Method:", ["Classical (OpenCV)", "AI-Based (Deep Learning)"])

# REMOVED "Video" from this list
input_type = st.sidebar.selectbox("Select Input:", ["Image", "Live Webcam"])

# --- MAIN LOGIC ---

# 1. IMAGE MODE
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_rgb, caption="Original", use_container_width=True)

        if st.button("Cartoonize"):
            with st.spinner("Processing..."):
                if processing_mode == "Classical (OpenCV)":
                    result = processors.classical_cartoonize_image(image)
                else:
                    result = processors.ai_cartoonize_image(image)

                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                with col2:
                    st.image(result_rgb, caption="Cartoonized", use_container_width=True)

                    # Allow saving the result
                    im_pil = Image.fromarray(result_rgb)
                    im_pil.save("final_output.png")
                    with open("final_output.png", "rb") as file:
                        btn = st.download_button(
                            label="Download Image",
                            data=file,
                            file_name="cartoonized.png",
                            mime="image/png"
                        )

# 2. WEBCAM MODE
elif input_type == "Live Webcam":
    st.warning("⚠️ Press 'Stop' to close the camera.")
    run_camera = st.checkbox("Start Webcam")

    frame_window = st.image([])

    if run_camera:
        camera = cv2.VideoCapture(0)

        while run_camera:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture image from camera.")
                break

            # Use the selected mode
            if processing_mode == "Classical (OpenCV)":
                processed_frame = processors.classical_cartoonize_image(frame)
            else:
                # The processor now handles the resizing for speed automatically
                processed_frame = processors.ai_cartoonize_image(frame)

            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_frame)

        camera.release()