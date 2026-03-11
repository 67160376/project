import streamlit as st
import pandas as pd
import joblib

# Load model และ encoder
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.title("📚 Comic Age Rating Predictor")

st.subheader("Comic Information")

# Numeric inputs
release_year = st.number_input("Release Year", 1900, 2030, 2000)
page_count = st.number_input("Page Count", 1, 2000, 100)
volume_count = st.number_input("Volume Count", 1, 100, 1)

# ใช้ค่าจาก encoder โดยตรง (แก้ unseen label error)
genre = st.selectbox(
    "Genre",
    encoders["Genre"].classes_
)

country = st.selectbox(
    "Country of Origin",
    encoders["Country of Origin"].classes_
)

format_type = st.selectbox(
    "Format",
    encoders["Format"].classes_
)

language = st.selectbox(
    "Language",
    encoders["Language"].classes_
)

status = st.selectbox(
    "Status",
    encoders["Status"].classes_
)

if st.button("🔍 Predict Age Rating"):

    # สร้าง dataframe input
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

    st.success(f"🎯 Predicted Age Rating: {rating}")

    st.subheader("Prediction Probability")

    # แสดง probability
    for i, label in enumerate(target_encoder.classes_):

        percent = round(proba[i] * 100, 2)

        st.write(f"{label}: {percent}%")

        st.progress(float(proba[i]))
