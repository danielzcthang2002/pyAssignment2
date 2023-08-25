import pandas as pd
import numpy as np
import streamlit as st
import joblib


# Load the data
LR_Model = joblib.load("dibetes.pkl")

st.title("Income Prediction")

# Create two number input fields
input1 = st.number_input("Enter Pregnancies", min_value=0)
input2 = st.number_input("Enter Glucose", min_value=0)
input3 = st.number_input("Enter BloodPressure", min_value=0)
input4 = st.number_input("Enter SkinThickness", min_value=0)
input5 = st.number_input("Enter Insulin", min_value=0)
input6 = st.number_input("Enter BMI", min_value=0)
input7 = st.number_input("Enter DiabetesPedigreeFunction", min_value=0.0, step=0.01)
input8 = st.number_input("Enter Age", min_value=0)
# Create a button
button_clicked = st.button("Predict Income")

# Check if the button is clicked
if button_clicked:
    user_input = [input1, input2, input3, input4,input5,input6,input7,input8]
    prediction = LR_Model.predict([user_input])
    if prediction[0] == 1:
        st.write('You have diabetes')
    else:
        st.write("You don't have diabetes")