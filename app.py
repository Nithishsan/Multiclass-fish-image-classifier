import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Streamlit page config
st.set_page_config(page_title="🐟 Fish or Not", page_icon="🐟", layout="centered")

# Load the trained binary model
MODEL_PATH = "models/fish_or_not_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Page title
st.title("🐠 Fish or Not Fish Classifier")
st.markdown("Upload an image and the model will determine whether it's a **fish** or **not**.")

# File uploader
uploaded = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB").resize((224, 224))
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        # Preprocess image
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)

        # Binary threshold
        if prediction[0][0] >= 0.5:
            label = "Not Fish"
            confidence = prediction[0][0]
        else:
            label = "Fish"
            confidence = 1 - prediction[0][0]

        # Show prediction
        st.markdown("### 📢 Prediction")
        st.success(f"**{label}** with **{confidence * 100:.2f}%** confidence")

        # Footer info
        st.markdown("---")
        st.markdown("🧠 **Model:** CNN  &nbsp;&nbsp;|&nbsp;&nbsp; 🎯 Input: `224x224`  &nbsp;&nbsp;|&nbsp;&nbsp; 🐟 Labels: Fish, Not Fish")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {str(e)}")

else:
    st.info("👈 Upload a fish or non-fish image to get started.")
