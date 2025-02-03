import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# Streamlit UI
st.title("Bank Subscription Prediction")
st.markdown("### Enter customer details to predict subscription outcome.")

# Only the features used in training (From your train.py)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                           'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'])
education = st.selectbox("Education", ['primary', 'secondary', 'tertiary'])
balance = st.number_input("Balance", min_value=-50000, max_value=100000, value=0)
day = st.number_input("Day", min_value=1, max_value=31, value=15)
duration = st.number_input("Duration", min_value=0, max_value=5000, value=200)
campaign = st.number_input("Campaign", min_value=1, max_value=100, value=3)
pdays = st.number_input("Pdays", min_value=-1, max_value=999, value=0)
previous = st.number_input("Previous", min_value=0, max_value=50, value=0)

# Create DataFrame for input
input_data = pd.DataFrame([[age, job, education, balance, day, duration, campaign, pdays, previous]],
                          columns=['age', 'job', 'education', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'])

# Encode categorical features exactly as in training
input_data = pd.get_dummies(input_data, drop_first=True)

# Ensure columns match model input
missing_cols = set(model.feature_names_in_) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0

input_data = input_data[model.feature_names_in_]  # Reorder columns

# Make Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_label = "Subscribed" if prediction[0] == 1 else "Not Subscribed"
    st.success(f"Prediction: {prediction_label}")
