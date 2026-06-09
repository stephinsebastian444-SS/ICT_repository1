import streamlit as st
import numpy as np
import pandas as pd
import pickle
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🩺 Diabetes Prediction System")
st.markdown(
    "Enter the patient details below to predict the likelihood of diabetes."
)

st.divider()
def yes_no_to_binary(value):
    return 1 if value == "Yes" else 0

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

hypertension = yes_no_to_binary(
    st.selectbox(
        "Hypertension",
        ["No", "Yes"]
    )
)

heart_disease = yes_no_to_binary(
    st.selectbox(
        "Heart Disease",
        ["No", "Yes"]
    )
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=24.0,
    step=0.1
)

HbA1c_level = st.number_input(
    "HbA1c Level",
    min_value=3.0,
    max_value=15.0,
    value=5.5,
    step=0.1
)

blood_glucose_level = st.number_input(
    "Blood Glucose Level",
    min_value=50,
    max_value=300,
    value=100
)


gender_male = yes_no_to_binary(
    st.selectbox(
        "Gender Male",
        ["No", "Yes"]
    )
)
st.subheader("Smoking History")

smoking_current = yes_no_to_binary(
    st.selectbox(
        "Current Smoker",
        ["No", "Yes"]
    )
)

smoking_ever = yes_no_to_binary(
    st.selectbox(
        "Ever Smoked",
        ["No", "Yes"]
    )
)

smoking_former = yes_no_to_binary(
    st.selectbox(
        "Former Smoker",
        ["No", "Yes"]
    )
)

smoking_never = yes_no_to_binary(
    st.selectbox(
        "Never Smoked",
        ["No", "Yes"]
    )
)

smoking_not_current = yes_no_to_binary(
    st.selectbox(
        "Not Currently Smoking",
        ["No", "Yes"]
    )
)

smoking_no_info = yes_no_to_binary(
    st.selectbox(
        "Smoking Info Not Available",
        ["No", "Yes"]
    )
)

st.divider()
if st.button("Predict Diabetes"):

    sample_data = [[
        age,
        hypertension,
        heart_disease,
        bmi,
        HbA1c_level,
        blood_glucose_level,
        gender_male,
        smoking_current,
        smoking_ever,
        smoking_former,
        smoking_never,
        smoking_not_current,
        smoking_no_info
    ]]
    sample_df = pd.DataFrame(sample_data)
    scaled_data = scaler.transform(sample_df)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Person is Likely Diabetic")
        st.write(f"Prediction Confidence: {probability * 100:.2f}%")

        st.markdown("### Recommendations")
        st.write("- Reduce sugary foods")
        st.write("- Exercise regularly")
        st.write("- Monitor blood sugar levels")
        st.write("- Increase vegetable intake")

    else:
        st.success("✅ Person is Likely Non-Diabetic")
        st.write(f"Prediction Confidence: {(1 - probability) * 100:.2f}%")

        st.markdown("### Health Tips")
        st.write("- Maintain a balanced diet")
        st.write("- Continue regular exercise")
        st.write("- Get periodic health checkups")
