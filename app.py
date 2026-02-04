import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# --------------------
# PAGE CONFIG
# --------------------
st.set_page_config(page_title="Flower Detection", page_icon="🌸")
st.title("🌸 Flower Detection App")

# --------------------
# LOAD MODEL
# --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("flower_model")

model = load_model()

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# --------------------
# IMAGE UPLOAD
# --------------------
uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

    # --------------------
    # PREPROCESS (MATCHES TRAINING)
    # --------------------
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # --------------------
    # PREDICT
    # --------------------
    predictions = model.predict(img_array)

    # --------------------
    # SHOW TOP 3 PREDICTIONS
    # --------------------
    st.subheader("Top Predictions")

    top_3_idx = np.argsort(predictions[0])[::-1][:3]
    for i in top_3_idx:
        st.write(f"{class_names[i]}: {predictions[0][i] * 100:.2f}%")

    # --------------------
    # FINAL RESULT
    # --------------------
    predicted_class = class_names[top_3_idx[0]]
    confidence = predictions[0][top_3_idx[0]] * 100

    st.success(f"🌼 Final Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
