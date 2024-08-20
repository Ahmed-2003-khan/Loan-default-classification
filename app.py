import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# Load model and feature names
with open('model1.pkl', 'rb') as file:
    model = pickle.load(file)

with open('feature_names.json', 'r') as file:
    feature_names = json.load(file)

# Function to convert age to categorical
def process_age(age):
    age_dict = {
        "age_<25": 0,
        "age_35-44": 0,
        "age_45-54": 0,
        "age_55-64": 0,
        "age_65-74": 0,
        "age_>74": 0
    }
    
    if age < 25:
        age_dict["age_<25"] = 1
    elif 35 <= age <= 44:
        age_dict["age_35-44"] = 1
    elif 45 <= age <= 54:
        age_dict["age_45-54"] = 1
    elif 55 <= age <= 64:
        age_dict["age_55-64"] = 1
    elif 65 <= age <= 74:
        age_dict["age_65-74"] = 1
    else:
        age_dict["age_>74"] = 1
    
    return age_dict

# Custom CSS for styling with purple background and better text visibility
st.markdown(
    """
    <style>
    .stApp {
        background-color: #7393B3;
        font-family: Arial, sans-serif;
    }
    .title {
        color: #333333;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    .form {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        max-width: 600px;
        margin: auto;
    }
    .form-input {
        margin-bottom: 20px;
    }
    label {
        color: #333333;
        font-weight: bold;
        margin-bottom: 5px;
        font-size: 16px;
    }
    input, select, textarea {
        background-color: #333333;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        padding: 10px;
    }
    .result-text {
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .slider-label {
        font-size: 20px;
        font-weight: bold;
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Title of the form
st.markdown("<div class='title'>Loan Default Prediction</div>", unsafe_allow_html=True)

# Form inputs using sliders
st.markdown("<div class='slider-label'>Upfront Charges</div>", unsafe_allow_html=True)
upfront_charges = st.slider("", min_value=0, max_value=10000, value=1000, step=100)
st.markdown("<div class='slider-label'>Rate of interest</div>", unsafe_allow_html=True)
rate_of_interest = st.slider("", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
st.markdown("<div class='slider-label'>Interest Rate Spread</div>", unsafe_allow_html=True)
interest_rate_spread = st.slider("", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st.markdown("<div class='slider-label'>Credit Score</div>", unsafe_allow_html=True)
credit_score = st.slider("", min_value=300, max_value=850, value=700, step=10)
st.markdown("<div class='slider-label'>Property Value</div>", unsafe_allow_html=True)
property_value = st.slider("", min_value=0, max_value=1000000, value=200000, step=10000)
st.markdown("<div class='slider-label'>Loan Amount</div>", unsafe_allow_html=True)
loan_amount = st.slider("", min_value=0, max_value=1000000, value=150000, step=10000)
st.markdown("<div class='slider-label'>Age</div>", unsafe_allow_html=True)
age = st.slider("", min_value=18, max_value=100, value=35, step=1)

# Process age
age_features = process_age(age)

# Prepare feature dictionary
input_data = {
    'upfront_charges': upfront_charges,
    'rate_of_interest': rate_of_interest,
    'interest_rate_spread': interest_rate_spread,
    'Credit_Score': credit_score,
    'property_value': property_value,
    'loan_amount': loan_amount
}

# Add age features
input_data.update(age_features)

# Convert to DataFrame
input_df = pd.DataFrame([input_data], columns=feature_names)

# Predict default probability
if st.button("Predict"):
    result = model.predict(input_df)
    
    if result == 1:
        st.error("There is a high chance of loan default", icon="⚠️")
        st.markdown("<div class='result-text' style='background-color: #ff4c4c; padding: 15px; border-radius: 5px; color: white;'>High probability of default. Caution advised!</div>", unsafe_allow_html=True)
    else:
        st.success("There is a low chance of loan default", icon="✅")
        st.markdown("<div class='result-text' style='background-color: #4caf50; padding: 15px; border-radius: 5px; color: white;'>Low probability of default. The loan is likely safe.</div>", unsafe_allow_html=True)

# End form container
st.markdown("</div>", unsafe_allow_html=True)
