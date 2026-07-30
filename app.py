import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import time

from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing.image import img_to_array

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="AI Vision Studio",
    page_icon="🖼️",
    layout="wide"
)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

image_model = load_model()

# ----------------------------------------------------
# Header
# ----------------------------------------------------
st.title("🖼️ AI Vision Studio")
st.subheader("Intelligent Image Recognition using MobileNetV2")

st.write("""
Recognize everyday objects using **Deep Learning** and **Transfer Learning**.

This application uses **MobileNetV2**, pretrained on the **ImageNet dataset**
containing over **1 million images** and **1000 object classes**.
""")

st.divider()

# ----------------------------------------------------
# Tabs
# ----------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Image Recognition",
    "🎯 Challenge Zone",
    "📚 How It Works",
    "🎓 Interview Corner",
    "ℹ️ About"
])

# ====================================================
# TAB 1
# ====================================================
with tab1:

    st.header("📤 Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns([2,1])

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:

            st.metric("Width", image.size[0])
            st.metric("Height", image.size[1])
            st.metric("Format", image.format)

        st.divider()

        if st.button("🔍 Analyze Image", use_container_width=True):

            with st.spinner("Analyzing Image..."):

                resized_image = image.resize((224,224))

                image_array = img_to_array(resized_image)

                image_batch = np.expand_dims(image_array, axis=0)

                processed_image = preprocess_input(image_batch)

                start = time.time()

                predictions = image_model.predict(
                    processed_image,
                    verbose=0
                )

                end = time.time()

                elapsed = (end-start)*1000

                decoded = decode_predictions(predictions, top=5)[0]

                st.success("Prediction Completed!")

                st.metric(
                    "🥇 Top Prediction",
                    decoded[0][1].replace("_"," ").title(),
                    f"{decoded[0][2]*100:.2f}% Confidence"
                )

                st.divider()

                st.subheader("📊 Top 5 Predictions")

                df = pd.DataFrame({
                    "Object":[x[1].replace("_"," ").title() for x in decoded],
                    "Confidence (%)":[round(x[2]*100,2) for x in decoded]
                })

                st.dataframe(df,use_container_width=True)

                st.subheader("Confidence Scores")

                for _,label,prob in decoded:

                    st.write(
                        f"**{label.replace('_',' ').title()}**"
                    )

                    st.progress(float(prob))

                    st.write(f"{prob*100:.2f}%")

                st.divider()

                c1,c2,c3,c4 = st.columns(4)

                c1.metric("⚡ Prediction Time",
                          f"{elapsed:.2f} ms")

                c2.metric("🧠 Model",
                          "MobileNetV2")

                c3.metric("📚 Dataset",
                          "ImageNet")

                c4.metric("🏷 Classes",
                          "1000")

# ====================================================
# TAB 2 - Challenge Zone
# ====================================================

with tab2:

    st.header("🎯 AI Challenge Zone")

    st.info(
        "Test the AI using different types of images and observe how the predictions change."
    )

    st.subheader("🏆 Challenges")

    challenge1, challenge2 = st.columns(2)

    with challenge1:

        st.checkbox("🐶 Upload a Dog Image")

        st.checkbox("🐱 Upload a Cat Image")

        st.checkbox("🚗 Upload a Car Image")

    with challenge2:

        st.checkbox("🐦 Upload a Bird Image")

        st.checkbox("🍔 Upload a Food Image")

        st.checkbox("🍼 Upload a Bottle Image")

    st.divider()

    st.subheader("🧪 AI Experiments")

    experiment = st.selectbox(
        "Choose an Experiment",
        (
            "Blurry Image",
            "Multiple Objects",
            "Dark Image",
            "Rotated Image",
            "Close-up Object",
        ),
    )

    if experiment == "Blurry Image":
        st.warning(
            """
            Upload a blurry image and compare the confidence score
            with a clear image.

            Observation:
            Blurry images usually reduce prediction confidence.
            """
        )

    elif experiment == "Multiple Objects":
        st.warning(
            """
            Upload an image containing multiple objects.

            Observe which object the AI focuses on first.
            """
        )

    elif experiment == "Dark Image":
        st.warning(
            """
            Upload a dark image.

            Low lighting may reduce prediction confidence.
            """
        )

    elif experiment == "Rotated Image":
        st.warning(
            """
            Rotate an image and compare predictions.

            Does the AI still recognize it?
            """
        )

    elif experiment == "Close-up Object":
        st.warning(
            """
            Try a close-up photo of an object.

            Compare it with a full image.
            """
        )

    st.divider()

    st.subheader("💡 What Did You Learn?")

    st.write(
        """
- AI predictions are probabilistic.
- Higher confidence does **not** always mean the prediction is correct.
- Image quality affects predictions.
- Multiple objects can confuse the model.
"""
    )

# ====================================================
# TAB 3 - How It Works
# ====================================================

with tab3:

    st.header("📚 How AI Recognizes an Image")

    st.success("Step 1️⃣  Upload an Image")

    st.write("The user uploads an image.")

    st.success("Step 2️⃣ Resize")

    st.write(
        "The image is resized to **224 × 224 pixels**, the required input size for MobileNetV2."
    )

    st.success("Step 3️⃣ Preprocessing")

    st.write(
        "`preprocess_input()` converts pixel values into the format expected by MobileNetV2."
    )

    st.success("Step 4️⃣ Prediction")

    st.write(
        "The preprocessed image is passed through the pretrained MobileNetV2 neural network."
    )

    st.success("Step 5️⃣ Probabilities")

    st.write(
        "The model predicts probabilities for **1000 ImageNet classes**."
    )

    st.success("Step 6️⃣ Final Output")

    st.write(
        "The class with the highest probability is displayed as the final prediction."
    )

    st.divider()

    st.subheader("Workflow")

    st.code(
"""
Image
   ↓
Resize (224×224)
   ↓
Preprocess
   ↓
MobileNetV2
   ↓
1000 Class Probabilities
   ↓
Top Prediction
"""
    )

# ====================================================
# TAB 4 - Interview Corner
# ====================================================

with tab4:

    st.header("🎓 Interview Corner")

    with st.expander("1. What is Transfer Learning?"):
        st.write(
            """
Transfer Learning means using a model that has already been trained
on a large dataset instead of training from scratch.
It saves time, data and computing resources.
"""
        )

    with st.expander("2. Why do we resize images to 224×224?"):
        st.write(
            """
MobileNetV2 accepts images of size 224×224.
Every image must be resized before prediction.
"""
        )

    with st.expander("3. What does preprocess_input() do?"):
        st.write(
            """
It converts pixel values into the format expected by MobileNetV2,
which improves prediction accuracy.
"""
        )

    with st.expander("4. Why does the model output probabilities?"):
        st.write(
            """
The model calculates the probability of every class and
selects the one with the highest probability.
"""
        )

    with st.expander("5. When can this model fail?"):
        st.write(
            """
It may perform poorly on blurry images,
medical images,
satellite images,
or objects not included in the ImageNet dataset.
"""
        )

# ====================================================
# TAB 5 - About
# ====================================================

with tab5:

    st.header("ℹ️ About This Project")

    st.write("""
### Project Name
AI Vision Studio

### Model
MobileNetV2

### Framework
TensorFlow / Keras

### Dataset
ImageNet

### Technique
Transfer Learning

### Total Classes
1000

### Objective
Recognize everyday objects from uploaded images using a pretrained
deep learning model.
""")

    st.divider()

    st.info(
        """
This application demonstrates how pretrained deep learning models
can recognize objects without requiring us to train a neural network
from scratch.
"""
    )

    st.divider()

    st.caption("👩‍💻 Developed by Sonali ")