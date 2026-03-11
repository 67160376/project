import streamlit as st
import pandas as pd
import joblib

# load model
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.title("📚 Comic Age Rating Predictor")

release_year = st.number_input("Release Year",1980,2025,2000)
page_count = st.number_input("Page Count",1,1000,30)
volume_count = st.number_input("Volume Count",1,50,1)

genre = st.selectbox("Genre",
["Action","Fantasy","Comedy","Drama"])

country = st.selectbox("Country of Origin",
["USA","Japan","Korea"])

format_type = st.selectbox("Format",
["Print","Digital"])

language = st.selectbox("Language",
["English","Japanese"])

status = st.selectbox("Status",
["Ongoing","Completed"])


if st.button("Predict Age Rating"):

    input_data = pd.DataFrame([{
        "Release Year":release_year,
        "Page Count":page_count,
        "Volume Count":volume_count,
        "Genre":genre,
        "Country of Origin":country,
        "Format":format_type,
        "Language":language,
        "Status":status
    }])


    for col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])


    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    rating = target_encoder.inverse_transform([pred])[0]

    st.success(f"Predicted Age Rating: {rating}")

    st.subheader("Prediction Probability")

    for i,label in enumerate(target_encoder.classes_):

        st.write(f"{label}: {round(proba[i]*100,2)}%")
        st.progress(float(proba[i]))
