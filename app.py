# streamlit_app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and preprocessing
with open("model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    preprocessor = data["preprocessor"]
    le_target = data["label_encoder"]
    features = data["features"]

st.set_page_config(page_title="Stress Level Predictor", layout="centered")
st.title("🧠 Predict Your Stress Category")
st.write("Fill out the following details to predict your stress level category.")

# Create input fields
input_data = {}

# Gender
input_data['Gender'] = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Other"])

# Your Major
input_data['Your major'] = st.selectbox("Your Major", ["Computer Science", "Biotech", "Mechanical", "EEE/ECE", "Other"])

# Diagnosis
input_data['Have you ever been diagnosed with a mental health condition by a professional (doctor, therapist, etc.)?'] = st.selectbox(
    "Diagnosed with mental health condition?", ["Yes", "No"])

# Treatment
input_data['Have you ever received treatment/support for a mental health problem?'] = st.selectbox(
    "Received mental health treatment/support?", ["Yes", "No"])

# Sleep Times
input_data['When have you usually gone to bed in the past month?'] = st.selectbox("Usual Bedtime", ["9pm-11pm", "11pm-1am", "1am-3am"])
input_data['How long has it taken you to fall asleep each night in the past month?'] = st.selectbox("Time to fall asleep", ["15 minutes", "30 minutes", "1 hour", "More time than 2 hours"])
input_data['What time have you usually gotten up in the morning in the past month?'] = st.selectbox("Wakeup time", ["before 8 am", "8 -10 am", "after 10 am"])
input_data['How many hours of actual sleep did you get on an average for the past month? (maybe different from the number of hours spent in bed)'] = st.selectbox("Avg. Sleep Hours", ["<5", "5-6", "6-7", "7-8", ">8"])

# Age
input_data['Age'] = st.selectbox("Age Group", ["18-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "65+"])

# Frequency-based questions
frequency_labels = [
    'Not during the past month', 'Less than once a week',
    'Once or twice a week', 'Three or more times a week']

frequency_mapping = {
    'Not during the past month': 0,
    'Less than once a week': 1,
    'Once or twice a week': 2,
    'Three or more times a week': 3
}

frequency_questions = [col for col in features if "During the past month" in col]
for col in frequency_questions:
    user_choice = st.selectbox(col.split("[")[-1].strip("]"), frequency_labels)
    input_data[col] = frequency_mapping[user_choice]

# Age mapping
age_mapping = {
    '18-20': 19, '21-25': 23, '26-30': 28, '31-35': 33, '36-40': 38,
    '41-45': 43, '46-50': 48, '51-55': 53, '56-60': 58, '61-65': 63, '65+': 68
}
input_data['Age'] = age_mapping[input_data['Age']]

# Create input DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess
X_processed = preprocessor.transform(input_df)

# Predict
if st.button("Predict Stress Level"):
    prediction = model.predict(X_processed)
    stress_label = le_target.inverse_transform(prediction)[0]
    st.success(f"🧠 Predicted Stress Level Category: **{stress_label.upper()}**")
