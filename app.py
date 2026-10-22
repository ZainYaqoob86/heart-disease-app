import streamlit as st
import pickle
import pandas as pd

with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("heart_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Heart Disease Prediction")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=["Female (0)", "Male (1)"])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

if st.button("Predict"):
    sex_val = 1 if "Male" in sex else 0

    input_df = pd.DataFrame([[age, sex_val, thalach, oldpeak]],
                            columns=["age", "sex", "thalach", "oldpeak"])

    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]

    if prediction == 1:
        st.error("⚠️ High risk of heart disease detected.")
    else:
        st.success("✅ Low risk of heart disease.")
        