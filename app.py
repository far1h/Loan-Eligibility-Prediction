import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle

# Load the model
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

# Web app
st.title('Loan Approval Prediction')
st.sidebar.header('Input Features')


# Sidebar inputs
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
married = st.sidebar.selectbox("Married", options=["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", options=["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", options=["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", options=["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", value=0)
loan_amount = st.sidebar.number_input("Loan Amount", value=0)
loan_amount_term = st.sidebar.number_input("Loan Amount Term", value=0)
credit_history = st.sidebar.selectbox("Credit History", options=["Good", "Bad"])
property_area = st.sidebar.selectbox("Property Area", options=["Rural", "Semiurban", "Urban"])

# Create a DataFrame with the specified columns
input_df = pd.DataFrame(columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                                  'Credit_History', 'Gender', 'Married', 'Dependents_0', 'Dependents_1',
                                  'Dependents_2', 'Dependents_3+', 'Education_Not Graduate', 'Self_Employed',
                                  'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban'])

# Add a new row with the input data
input_df.loc[0] = [applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                   1 if credit_history == "Good" else 0,
                   1 if gender == "Male" else 0,
                   1 if married == "Yes" else 0,
                   1 if dependents == "0" else 0,
                   1 if dependents == "1" else 0,
                   1 if dependents == "2" else 0,
                   1 if dependents == "3+" else 0,
                   0 if education == "Graduate" else 1,
                   1 if self_employed == "Yes" else 0,
                   1 if property_area == "Rural" else 0,
                   1 if property_area == "Semiurban" else 0,
                   1 if property_area == "Urban" else 0
                   ]

# Display input data
st.subheader("Input Data Summary")
st.write("Gender:", gender)
st.write("Married:", married)
st.write("Dependents:", dependents)
st.write("Education:", education)
st.write("Self Employed:", self_employed)
st.write("Applicant Income:", applicant_income)
st.write("Coapplicant Income:", coapplicant_income)
st.write("Loan Amount:", loan_amount)
st.write("Loan Amount Term:", loan_amount_term)
st.write("Credit History:", credit_history)
st.write("Property Area:", property_area)

# Make predictions
prediction = model.predict(input_df)

# Display prediction
st.subheader("Prediction")
if prediction[0] == 1:
    st.success("Loan Approved")
else:
    st.error("Loan Rejected")

# Display model information
st.subheader("Model Information")
st.write("Algorithm used: Random Forest Classifier")
st.write("Accuracy: 85%")
st.write("Creator: John Doe")
st.write("Date created: January 1, 2023")
