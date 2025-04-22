import streamlit as st
import pandas as pd
import pickle

# Load models and encoders
xgb = pickle.load(open("xgb_model.pkl", "rb"))
gender_encoder = pickle.load(open("gender_encode.pkl", "rb"))
previous_loan_encoder = pickle.load(open("previous_loan_encode.pkl", "rb"))
person_education_encoder = pickle.load(open("person_education_encode.pkl", "rb"))
person_home_ownership_encoder = pickle.load(open("person_home_ownership_encode.pkl", "rb"))
loan_intent_encoder = pickle.load(open("loan_intent_encode.pkl", "rb"))

scalers = {}
for col in ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate']:
    scalers[col] = pickle.load(open(f"{col}_scaler.pkl", "rb"))

st.title("Loan Approval Prediction App")

st.markdown("""
Masukkan informasi di bawah ini untuk memprediksi apakah pemohon akan mendapatkan persetujuan pinjaman.
""")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Income", min_value=0)
education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
emp_exp = st.number_input("Employment exp (Years)", min_value=0.0, step=0.1)
home = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage", "Other"])
intent = st.selectbox("Loan Intent", ["Debt Consolidation", "Home Improvement", "Education", "Medical", "Personal", "Venture"])
amount = st.number_input("Loan Amount", min_value=0.0)
rate = st.number_input("Loan Interest Rate (%)", min_value=0.0)
default = st.selectbox("Previous Loan Defaults", ["Yes", "No"])

if st.button("Predict"):
    # Create dataframe
    df = pd.DataFrame({
        "person_gender": [gender_encoder[gender]],
        "person_age": [scalers['person_age'].transform([[age]])[0][0]],
        "person_income": [scalers['person_income'].transform([[income]])[0][0]],
        "person_education": [person_education_encoder['person_education'][education]],
        "person_emp_exp": [scalers['person_emp_exp'].transform([[emp_exp]])[0][0]],
        "loan_amnt": [scalers['loan_amnt'].transform([[amount]])[0][0]],
        "loan_int_rate": [scalers['loan_int_rate'].transform([[rate]])[0][0]],
        "previous_loan_defaults_on_file": [previous_loan_encoder[default]]
    })

    # One-hot encoding
    home_encoded = person_home_ownership_encoder.transform([[home]]).toarray()
    home_encoded_df = pd.DataFrame(home_encoded, columns=person_home_ownership_encoder.get_feature_names_out())

    intent_encoded = loan_intent_encoder.transform([[intent]]).toarray()
    intent_encoded_df = pd.DataFrame(intent_encoded, columns=loan_intent_encoder.get_feature_names_out())

    final_df = pd.concat([df.reset_index(drop=True), home_encoded_df, intent_encoded_df], axis=1)

    # Use the xgb model for prediction
    prediction = xgb.predict(final_df)[0]
    label = "Approved" if prediction == 1 else "Rejected"
    st.subheader(f"Loan Status Prediction: {label}")
