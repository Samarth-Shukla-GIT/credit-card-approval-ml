# app.py
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("credit_model.pkl", "rb") as f:
    model = pickle.load(f)
  

st.set_page_config(page_title="Credit Card Approval Prediction", layout="centered")

st.title("💳 Credit Card Approval Prediction")

# Input fields
gender = st.selectbox("Gender", ["M", "F"])
own_car = st.selectbox("Own Car", ["Y", "N"])
own_realty = st.selectbox("Own Realty", ["Y", "N"])
children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
income = st.number_input("Annual Income", min_value=10000.0, step=500.0)
income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner"])
education = st.selectbox("Education", ["Secondary / secondary special", "Higher education", "Incomplete higher"])
family_status = st.selectbox("Family Status", ["Single / not married", "Married", "Civil marriage", "Separated"])
housing = st.selectbox("Housing Type", ["House / apartment", "Rented apartment"])
mobile = st.selectbox("Has Mobile?", [1, 0])
work_phone = st.selectbox("Has Work Phone?", [1, 0])
phone = st.selectbox("Has Phone?", [1, 0])
email = st.selectbox("Has Email?", [1, 0])
occupation = st.text_input("Occupation", "Laborers")
fam_members = st.number_input("Family Members", min_value=1.0, step=0.5)
years_birth = st.number_input("Age (Years)", min_value=18, max_value=100)
years_employed = st.number_input("Years Employed", min_value=0, max_value=60)

# Predict button
if st.button("Predict Approval"):
    input_df = pd.DataFrame([{
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": children,
        "AMT_INCOME_TOTAL": income,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing,
        "FLAG_MOBIL": mobile,
        "FLAG_WORK_PHONE": work_phone,
        "FLAG_PHONE": phone,
        "FLAG_EMAIL": email,
        "OCCUPATION_TYPE": occupation,
        "CNT_FAM_MEMBERS": fam_members,
        "YEARS_BIRTH": years_birth,
        "YEARS_EMPLOYED": years_employed,
    }])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Credit Card Approved! (Confidence: {proba:.2f})")
    else:
        st.error(f"❌ Credit Card Not Approved (Confidence: {proba:.2f})")

