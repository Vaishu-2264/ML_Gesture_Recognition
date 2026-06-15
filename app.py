import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import pandas as pd

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

st.markdown("""
<style>

/* Background */
.stApp{
    background:#F5F7FB;
}

/* Container */
.block-container{
    max-width:1500px;
    padding-top:0rem;
}

/* Hero Section */
.hero{
    background:linear-gradient(
        135deg,
        #07152B,
        #02204F,
        #010D1F
    );

    padding:0px 30px;
    border-radius:0 0 25px 25px;

    text-align:center;
    color:white;

    margin-bottom:20px;
}

/* Badge */
.badge{
    display:inline-block;

    padding:8px 18px;

    background:#0E2D63;

    border-radius:10px;

    font-size:14px;

    margin-bottom:20px;
}

/* Heading */
.hero h1{
    font-size:60px;
    font-weight:700;
    line-height:1.2;
}

.hero span{
    color:#3B82F6;
}

/* Paragraph */
.hero p{
    font-size:22px;
    color:#D1D5DB;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
    justify-content:center;
    gap:20px;
    background:white;
    border-radius:12px;
    padding:10px;
}

.stTabs [data-baseweb="tab"]{
    font-size:18px;
    font-weight:600;
}

/* Cards */
.feature-card{
    background:white;
    padding:25px;
    border-radius:18px;

    box-shadow:0 4px 20px rgba(0,0,0,0.08);

    text-align:center;

    height:250px;
}

.feature-title{
    font-size:22px;
    font-weight:700;
    color:#111827;
}

.feature-text{
    color:#6B7280;
}

.section-card{
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0 4px 15px rgba(0,0,0,0.08);
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

data = joblib.load("Gesture_recognition_prod_file.pkl")

model = data["model"]
scaler = data["scaler"]

# ---------------- TITLE ----------------

st.markdown("""
<style>
.block-container {
    padding-top: 2.2rem;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">

<div class="badge">
🚀 MACHINE LEARNING POWERED
</div>

<h1>
American Sign Language (ASL)<br>
<span>Recognition System</span>
</h1>

<p>
Recognize ASL hand gestures from images and convert them into
English alphabets and words with high accuracy.
</p>

</div>
""", unsafe_allow_html=True)

# ---------------- TABS ----------------

tab1, tab2, tab3 = st.tabs(
    ["🏠 Home", "📷 Prediction", "ℹ️ About Project"]
)

# ====================================================
# HOME TAB
# ====================================================

with tab1:

    col1,col2 = st.columns([1,1])

    with col1:
        st.markdown("""
        ### Project Overview

        The ASL Recognition System is a Machine Learning application
        that recognizes sign language alphabets from images.

        Multiple predictions can be combined into complete words.
        """)
    with col2:
        st.markdown("""
        ### Workflow
        """)
        st.image(
            "Workflow.png",
            width=500
        )

    st.markdown("## Key Features")

    col1,col2,col3,col4,col5 = st.columns(5)

    cards = [
    ("🤟","ASL Recognition",
    "Recognizes ASL alphabet gestures A-Z."),
    ("🔤","Text Conversion",
    "Converts gestures into text."),
    ("📂","Multiple Uploads",
    "Supports multiple image uploads."),
    ("⚡","Fast & Accurate",
    "Machine learning predictions."),
    ("🌐","Web Application",
    "Built using Streamlit.")
    ]

    for col,data in zip([col1,col2,col3,col4,col5],cards):

        with col:
            st.markdown(f"""
            <div class="feature-card">
                <h1>{data[0]}</h1>
                <div class="feature-title">{data[1]}</div>
                <br>
                <div class="feature-text">{data[2]}</div>
            </div>
            """,
            unsafe_allow_html=True)
    


# ====================================================
# PREDICTION TAB
# ====================================================

with tab2:

    st.header("Gesture Prediction")

    st.markdown("""
    Upload the files in a sequence one after other.""")

    left_col, right_col = st.columns([1, 2])

    with left_col:

        uploaded_files = st.file_uploader(
            "Upload Gesture Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files:

            word = ""

            predictions = []

            for uploaded_file in uploaded_files:

                image = Image.open(uploaded_file)

                img = np.array(image)

                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                img = cv2.resize(img, (28, 28))

                img_flat = img.flatten().reshape(1, -1)

                img_scaled = scaler.transform(img_flat)

                pred = model.predict(img_scaled)

                letter = chr(pred[0] + 65)

                word += letter

                predictions.append((image, letter))

            st.markdown("### Predicted Word")

            highlighted_word = f"""
            <div style="
                background: linear-gradient(90deg, #2563EB, #7C3AED);
                padding: 10px;
                border-radius: 10px;
                text-align: center;
                margin-top: 10px;
                margin-bottom: 20px;
            ">
                <span style="
                    font-size: 42px;
                    font-weight: bold;
                    color: white;
                    letter-spacing: 3px;
                ">
                    {word.capitalize()}
                </span>
            </div>
            """

            st.markdown(highlighted_word, unsafe_allow_html=True)

    with right_col:

        if uploaded_files:

            st.markdown("### Prediction Details")

            cols = st.columns(min(4, len(predictions)))

            for idx, (image, letter) in enumerate(predictions):

                with cols[idx % 4]:

                    st.image(
                        image,
                        width=100
                    )

                    st.metric(
                        "Letter",
                        letter
                    )

# ====================================================
# ABOUT TAB
# ====================================================

with tab3:

    st.subheader("About the Project")

    st.write("""
    This project demonstrates image classification using Machine Learning
    techniques to recognize American Sign Language (ASL) alphabet gestures.

    The uploaded gesture images are preprocessed, converted into pixel-based
    features, and passed to a trained classification model to predict the
    corresponding alphabet.
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("""ASL Gesture Dataset Overview""")
    st.image(
            "amer_sign2.png",
            width=1000
        )

    col1, col2 = st.columns(2)

    # Dataset Details
    with col1:

        st.markdown("""
        <div class="section-card">
        <h3>📂 Dataset Details</h3>

        <ul>
        <li><b>Dataset:</b> Sign language MNIST</li>
        <li><b>Classes:</b> 0-24 (except 9)</li>
        <li><b>Image Size:</b> 28 × 28 Pixels</li>
        <li><b>Features:</b> Pixel Values</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">

        <h3>🛠️ Tools & Technologies Used</h3>

        • Python<br>
        • NumPy<br>
        • OpenCV<br>
        • Scikit-Learn<br>
        • Joblib<br>
        • Pillow (PIL)<br>
        • Streamlit

        </div>
        """, unsafe_allow_html=True)


    st.markdown("""
        <div class="info-card">
        <h3>📊 Model Accuracy</h3>
        </div>
        """, unsafe_allow_html=True)

    accuracy_df = pd.DataFrame({
        "Model": ["KNN", "SVM", "Logistic Regression"],
        "Accuracy (%)": [95, 98, 99]
    })

    st.dataframe(
        accuracy_df,
        hide_index=True,
        width=400
    )

    st.info(
        "Logistic Regression was selected as the final model because it achieved the highest accuracy among all evaluated models."
    )