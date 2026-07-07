import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from pathlib import Path
from tensorflow.keras.preprocessing.image import img_to_array


# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="Chest X-Ray Pneumonia AI",
    page_icon="🫁",
    layout="wide"
)


# =====================================
# Model Path
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_PATH = (
    BASE_DIR
    / "Models"
    / "best_densenet121_pneumonia_final.keras"
)



# =====================================
# Load Model
# =====================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        st.error(
            f"Model not found:\n{MODEL_PATH}"
        )

        st.stop()


    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model


    except Exception as e:

        st.error(
            "Model loading failed"
        )

        st.exception(e)

        st.stop()



model = load_model()



# =====================================
# Check Model Layers
# =====================================

# برای پیدا کردن آخرین Conv Layer در DenseNet121

possible_layers = [
    layer.name 
    for layer in model.layers
]


last_conv_layer_name = None


for name in reversed(possible_layers):

    if "concat" in name or "conv" in name:

        last_conv_layer_name = name

        break



if last_conv_layer_name is None:

    st.error(
        "No convolution layer found for Grad-CAM"
    )

    st.stop()



# =====================================
# Grad-CAM Model
# =====================================


grad_model = tf.keras.models.Model(

    inputs=model.input,

    outputs=[
        model.get_layer(
            last_conv_layer_name
        ).output,

        model.output
    ]
)



# =====================================
# Grad-CAM Function
# =====================================


def generate_gradcam(img_array):


    with tf.GradientTape() as tape:


        conv_outputs, predictions = grad_model(
            img_array
        )


        loss = predictions[:,0]



    grads = tape.gradient(
        loss,
        conv_outputs
    )



    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )



    conv_outputs = conv_outputs[0]



    heatmap = conv_outputs @ pooled_grads[...,None]


    heatmap = tf.squeeze(
        heatmap
    )



    heatmap = tf.maximum(
        heatmap,
        0
    )



    max_value = tf.reduce_max(
        heatmap
    )


    if max_value != 0:

        heatmap /= max_value



    return heatmap.numpy()



# =====================================
# Prediction
# =====================================


def predict_xray(image):


    img = image.resize(
        (224,224)
    )


    img = img.convert(
        "RGB"
    )



    img_array = img_to_array(
        img
    )



    img_array = img_array / 255.0



    input_image = np.expand_dims(
        img_array,
        axis=0
    )



    prediction = model.predict(
        input_image
    )[0][0]



    if prediction >= 0.5:

        label = "PNEUMONIA"

        confidence = prediction


    else:

        label = "NORMAL"

        confidence = 1 - prediction



    heatmap = generate_gradcam(
        input_image
    )



    heatmap = cv2.resize(
        heatmap,
        (224,224)
    )



    original = np.array(
        img
    )



    heatmap_color = cv2.applyColorMap(

        np.uint8(
            255 * heatmap
        ),

        cv2.COLORMAP_JET

    )



    overlay = cv2.addWeighted(

        original,

        0.6,

        heatmap_color,

        0.4,

        0

    )



    return (
        label,
        confidence,
        overlay
    )



# =====================================
# UI
# =====================================


st.title(
    "🫁 Chest X-Ray Pneumonia AI"
)


st.write(
"""
Deep Learning application using DenseNet121
for pneumonia detection from chest X-Ray images.

Includes Grad-CAM explainability.
"""
)



uploaded_file = st.file_uploader(

    "Upload Chest X-Ray Image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]

)



if uploaded_file:


    image = Image.open(
        uploaded_file
    )



    st.subheader(
        "Uploaded X-Ray"
    )



    st.image(
        image,
        width=400
    )



    if st.button(
        "Analyze Image"
    ):


        with st.spinner(
            "AI is analyzing..."
        ):


            label, confidence, overlay = predict_xray(
                image
            )



        st.success(
            "Analysis Complete"
        )



        col1, col2 = st.columns(2)



        with col1:


            st.subheader(
                "Diagnosis"
            )


            if label == "PNEUMONIA":

                st.error(
                    label
                )

            else:

                st.success(
                    label
                )



            st.metric(

                "Confidence",

                f"{confidence*100:.2f}%"

            )



        with col2:


            st.subheader(
                "Grad-CAM Explanation"
            )


            st.image(
                overlay,
                width=400
            )



        st.info(
"""
Grad-CAM highlights regions
that influenced the AI prediction.
"""
        )
