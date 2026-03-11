import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------
# Load model
# -----------------------

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# -----------------------
# Page Config
# -----------------------

st.set_page_config(
    page_title="Comic Age Rating Predictor",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Comic Age Rating Prediction System")

st.markdown("Machine Learning model for predicting comic age ratings.")

# -----------------------
# Model Info
# -----------------------

st.sidebar.header("Model Information")

st.sidebar.write("Model Used:")
st.sidebar.success("Logistic Regression")

st.sidebar.write("Accuracy:")
st.sidebar.info("~82% (example)")

# -----------------------
# Input Section
# -----------------------

st.subheader("Comic Information")

col1, col2 = st.columns(2)

with col1:
    release_year = st.number_input("Release Year", 1900, 2030, 2000)
    page_count = st.number_input("Page Count", 1, 2000, 100)
    volume_count = st.number_input("Volume Count", 1, 100, 1)
    genre = st.selectbox("Genre", encoders["Genre"].classes_)

with col2:
    country = st.selectbox("Country of Origin", encoders["Country of Origin"].classes_)
    format_type = st.selectbox("Format", encoders["Format"].classes_)
    language = st.selectbox("Language", encoders["Language"].classes_)
    status = st.selectbox("Status", encoders["Status"].classes_)

# -----------------------
# Prediction
# -----------------------

if st.button("🔍 Predict Age Rating"):

    input_data = pd.DataFrame([{
        "Release Year": release_year,
        "Page Count": page_count,
        "Volume Count": volume_count,
        "Genre": genre,
        "Country of Origin": country,
        "Format": format_type,
        "Language": language,
        "Status": status
    }])

    # Encode categorical columns
    for col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

    # Prediction
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    rating = target_encoder.inverse_transform([pred])[0]

    # -----------------------
    # Result
    # -----------------------

    st.success(f"🎯 Predicted Age Rating: **{rating}**")

    st.subheader("Prediction Probability")

    prob_df = pd.DataFrame({
        "Rating": target_encoder.classes_,
        "Probability": proba
    })

    st.bar_chart(prob_df.set_index("Rating"))

    # Progress bars
    for i, label in enumerate(target_encoder.classes_):
        percent = round(proba[i] * 100, 2)
        st.write(f"{label}: {percent}%")
        st.progress(float(proba[i]))

# -----------------------
# Footer
# -----------------------

st.markdown("---")
st.caption("Machine Learning Project | Comic Rating Prediction")
