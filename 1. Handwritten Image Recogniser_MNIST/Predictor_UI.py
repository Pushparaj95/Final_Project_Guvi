import torch
from torchvision import transforms
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
from Model_class import ConvNet
import Model_class as mc

# Set page configuration
st.set_page_config(
    page_title="Drawing Pad",
    layout="wide",  # Use wide layout for better space utilization
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;  /* Extend the container width */
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "canvas_data" not in st.session_state:
    st.session_state["canvas_data"] = None
if "bg_image" not in st.session_state:
    st.session_state["bg_image"] = None

# Title
st.markdown("<h1 style='text-align: center; color: red;'>✏️ HANDWRITEN NUMBER RECOGNISER</h1>", unsafe_allow_html=True)

# Layout: Tools and Canvas
with st.container():
    col1, col2 = st.columns([1, 3])  # Allocate more space for the canvas

    # Tools column (left)
    with col1:
        st.markdown("### Tools")
        
        st.markdown("##### Brush Size")
        stroke_width = st.slider("Brush Size", 25, 30, 30)
        
        st.markdown("##### Brush Color")
        stroke_color = st.color_picker("Brush Color", "#000000")
        
        st.markdown("##### Background Color")
        bg_color = st.color_picker("Background Color", "#FFFFFF")
        
        st.markdown("##### Upload Image (optional)")
        uploaded_file = st.file_uploader(
            "",
            type=["png", "jpg", "jpeg"],
            key="file_uploader",
            help="Upload an image file to predict.",
            label_visibility="hidden"
        )
        if uploaded_file is not None:
            st.session_state["bg_image"] = uploaded_file
            st.write("Image uploaded successfully!")

    # Canvas section (centered)
    with col2:
        canvas_size = 800  # Set a larger width for the canvas
        background_image = None
        if st.session_state["bg_image"] is not None:
            uploaded_file = st.session_state["bg_image"]
            background_image = Image.open(uploaded_file).convert("RGBA")
            background_image = background_image.resize((canvas_size, 600))
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color if background_image is None else None,
            background_image=background_image,
            height=600,
            width=canvas_size,
            drawing_mode="freedraw",
            key="canvas_main" if st.session_state["canvas_data"] is None else "canvas_drawn",
        )

# Save canvas data
if canvas_result.image_data is not None:
    st.session_state["canvas_data"] = canvas_result.image_data

# Action buttons in the last column
if st.session_state["canvas_data"] is not None:
    st.markdown("---")
    with col2:  # Move the buttons into col2 (the larger column for canvas)
        col_analysis, col_clear, col_export = st.columns([1, 1, 1])

        with col_analysis:
            if st.button("🔍 Show Analysis"):
                canvas_data = st.session_state["canvas_data"]
                if canvas_data is not None:
                    # Image for downloading
                    rgba_image = Image.fromarray(canvas_data.astype("uint8"), "RGBA")
                    bg_rgb = tuple(int(bg_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                    background = Image.new("RGBA", rgba_image.size, bg_rgb + (255,))
                    final_image = Image.alpha_composite(background, rgba_image).convert("RGB")

                    # Image for Model
                    combined_image = Image.alpha_composite(background, rgba_image).convert("L")
                    tensor_image =  mc.preprocess_canvas_image(combined_image)
                    
                    if "model" not in st.session_state:
                        model = ConvNet()
                        model.load_state_dict(torch.load('mnist_model.pth'))
                        model.eval()
                        st.session_state["model"] = model

                    # Make prediction
                    with torch.no_grad():
                        output = st.session_state["model"](tensor_image)
                        _, predicted = torch.max(output, 1)
                        confidence = torch.nn.functional.softmax(output, dim=1)[0]
                        confidence_score = confidence[predicted].item() * 100

                    # Show results
                    st.write(f"Predicted Number: {predicted.item()}")
                    st.write(f"Confidence: {confidence_score:.2f}%")
        
        with col_clear:
            if st.button("🔄 Clear Canvas"):
                st.session_state["canvas_data"] = None
                st.session_state["bg_image"] = None

        with col_export:
            canvas_data = st.session_state["canvas_data"]
            if canvas_data is not None:
                rgba_image = Image.fromarray(canvas_data.astype("uint8"), "RGBA")
                bg_rgb = tuple(int(bg_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                background = Image.new("RGBA", rgba_image.size, bg_rgb + (255,))
                final_image = Image.alpha_composite(background, rgba_image).convert("RGB")
                
                buffer = io.BytesIO()
                final_image.save(buffer, format="PNG")
                buffer.seek(0)
                
                st.download_button(
                    label="📥 Export Drawing",
                    data=buffer,
                    file_name="drawing.png",
                    mime="image/png",
                )
