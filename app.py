import streamlit as st 
import tensorflow as tf 
import numpy as np 
from PIL import Image 
 
st.set_page_config(page_title="Brain Tumor Detection", layout="centered") 
 
@st.cache_resource 
def load_model(): 
    return tf.keras.models.load_model("Mai10.h5") 
 
model = load_model() 
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary'] 
 
tumor_info = { 
    "glioma": "A glioma is a type of tumor that occurs in the brain and spinal cord, originating from glial cells that support neurons.", 
    "meningioma": "A meningioma is a tumor that forms on membranes covering the brain and spinal cord, usually benign but can cause pressure effects.", 
    "no_tumor": "No tumor detected. The brain appears normal with no signs of abnormal growth.", 
    "pituitary": "A pituitary tumor forms in the pituitary gland, affecting hormone production and bodily functions." 
} 
 
st.markdown( 
    """ 
    <h1 style='text-align: center; color: white; background-color: #1E3A8A; 
    padding: 15px; border-radius: 10px;'> Brain Tumor Detection from MRI Images</h1> 
    """, 
    unsafe_allow_html=True 
) 
 
st.markdown( 
    """ 
    <style> 
    div.stButton > button:first-child { 
        background-color: #1E3A8A; 
        color: white; 
        border-radius: 8px; 
        padding: 8px 16px; 
        font-size: 16px; 
        border: none; 
        transition: all 0.3s; 
    } 
    div.stButton > button:hover { 
        background-color: #0f2573; 
        transform: scale(1.05); 
    } 
 
    @media (prefers-color-scheme: dark) { 
        .tumor-box { background-color: #1E293B !important; color: #E2E8F0 !important; border: 1px solid #3B82F6 !important; } 
        .highlight { background-color: #2563EB !important; color: #FFFFFF !important; border: 2px solid #93C5FD !important; } 
    } 
    @media (prefers-color-scheme: light) { 
        .tumor-box { background-color: #E0E7FF !important; color: #1E293B !important; border: 1px solid #1E3A8A !important; } 
        .highlight { background-color: #1E3A8A !important; color: #FFFFFF !important; border: 2px solid #93C5FD !important; } 
    } 
    </style> 
    """, 
    unsafe_allow_html=True 
) 
 
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"]) 
 
if uploaded_file is not None: 
    image_input = Image.open(uploaded_file).convert("RGB") 
    img_resized = image_input.resize((256, 256)) 
    img_array = np.array(img_resized, dtype=np.float32) 
    img_array = np.expand_dims(img_array, axis=0) 
 
    prediction = model.predict(img_array) 
    predicted_index = int(np.argmax(prediction[0])) 
    confidence = float(np.max(prediction[0]) * 100) 
    predicted_label = class_labels[predicted_index] 
 
    st.session_state['predicted_label'] = predicted_label 
    st.session_state['prediction_probs'] = prediction[0].tolist() 
    st.session_state['confidence'] = confidence 
 
    st.image(image_input, caption="Uploaded MRI Image", use_container_width=True) 
    st.success(f" Prediction: {predicted_label}") 
    st.info(f" Confidence: {confidence:.2f}%") 
 
if st.button(" Show tumor type info"): 
    if 'predicted_label' not in st.session_state: 
        st.warning("Please upload an image first so the model can predict the tumor type.") 
    else: 
        predicted_label = st.session_state['predicted_label'] 
        st.markdown("###  Tumor Type Information") 
        for label, description in tumor_info.items(): 
            if label == predicted_label: 
                st.markdown( 
                    f""" 
                    <div class="tumor-box highlight" style=" 
                        padding:10px; 
                        border-radius:10px; 
                        margin-bottom:10px; 
                        font-size:15px;"> 
                        <b> {label.upper()}</b>: {description}
                    </div> 
                    """, 
unsafe_allow_html=True 
                ) 
            else: 
                st.markdown( 
                    f""" 
                    <div class="tumor-box" style=" 
                        padding:8px; 
                        border-radius:10px; 
                        margin-bottom:8px; 
                        font-size:15px;"> 
                        <b>{label.upper()}</b>: {description} 
                    </div> 
                    """, 
                    unsafe_allow_html=True 
                ) 
 
st.divider() 
st.markdown( 
    """ 
    <div style="text-align:center; color:gray; font-size:16px;"> 
        <p> Have questions or feedback about this project?<br> 
        We're always happy to hear from you!</p> 
        <p> <b>Contact us:</b> <u>duaa.reham.mai.team@gmail.com</u></p> 
    </div> 
    """, 
    unsafe_allow_html=True 
)