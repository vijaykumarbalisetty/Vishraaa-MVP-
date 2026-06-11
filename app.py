import streamlit as st
import joblib

model = joblib.load("model.pkl")

le_occasion = joblib.load("occasion.pkl")
le_color = joblib.load("color.pkl")
le_style = joblib.load("style.pkl")
le_target = joblib.load("target.pkl")

st.title("Vishraa Fashion Recommender")

occasion = st.selectbox(
    "Select Occasion",
    le_occasion.classes_
)

color = st.selectbox(
    "Select Color",
    le_color.classes_
)

style = st.selectbox(
    "Select Style",
    le_style.classes_
)

if st.button("Recommend Outfit"):

    occ = le_occasion.transform([occasion])[0]
    col = le_color.transform([color])[0]
    sty = le_style.transform([style])[0]

    prediction = model.predict([[occ, col, sty]])

    outfit = le_target.inverse_transform(prediction)[0]

    st.success(f"Recommended Outfit: {outfit}")