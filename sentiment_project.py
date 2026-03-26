import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

# Cleaning function
def mycleaning(doc):
    return re.sub("[^a-zA-Z ]", "", str(doc)).lower()

# Load model AND vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")   # 🔴 IMPORTANT

st.set_page_config(layout='wide')

st.markdown("""
    <div style="
        background: linear-gradient(90deg, #ffff00, #2E7D32);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    ">
        <h1 style="
            color: purple;
            font-size: 40px;
            margin: 0;
        ">
            Food Sentiment Analysis
        </h1>
    </div>
""", unsafe_allow_html=True)

st.sidebar.image("flag.jpg")

st.sidebar.title("About Project")
st.sidebar.write("Prediction of Sentiment Neg or Pos for a food review")

st.sidebar.title("Contact us 📞")
st.sidebar.write("9999999999")

st.sidebar.title("About us👥")
st.sidebar.write("We are a group of AI Engineers at DUCAT")

st.write("\n")
st.write("#### Enter Review")

sample = st.text_input("")

if st.button("Predict"):
    if sample.strip() != "":
        clean = mycleaning(sample)
        vec = vectorizer.transform([clean])

        pred = model.predict(vec)
        prob = model.predict_proba(vec)

        if pred[0] == 0:
            st.write("Neg 👎")
            st.write(f"Confidence Score : {prob[0][0]:.2f}")
        else:
            st.write("Pos 👍")
            st.write(f"Confidence Score : {prob[0][1]:.2f}")
            st.balloons()
    else:
        st.warning("Please enter a review")

# Bulk Prediction
st.write("#### Bulk Prediction")

file = st.file_uploader("select file", type=["csv", "txt"])

if file:
    df = pd.read_csv(file, names=["Review"])
    placeholder = st.empty()
    placeholder.dataframe(df)

    if st.button("Predict", key="b2"):
        # Clean text
        corpus = df["Review"].apply(mycleaning)

        # Transform
        vec = vectorizer.transform(corpus)

        pred = model.predict(vec)
        prob = np.max(model.predict_proba(vec), axis=1)

        df['Sentiment'] = pred
        df['Confidence'] = prob

        df['Sentiment'] = df['Sentiment'].map({0: 'Neg 👎', 1: 'Pos 👍'})

        placeholder.dataframe(df)

import os

base_path = os.getcwd()

model_path = os.path.join(base_path, "sentiment_model.pkl")
vectorizer_path = os.path.join(base_path, "vectorizer.pkl")

st.write("Trying to load from:", vectorizer_path)

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)        