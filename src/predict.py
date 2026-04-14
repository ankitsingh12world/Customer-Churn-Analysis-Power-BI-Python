import streamlit as st
import pandas as pd
import joblib

# Load model & tools
model = joblib.load("../models/churn_model.pkl")
scaler = joblib.load("../models/Scaler.pkl")
ordinal_encoder = joblib.load("../models/Ordinal_Encoder.pkl")
onehot_encoder = joblib.load("../models/One_Hot_Encoder.pkl")

st.title("Customer Churn Prediction")

# User inputs
age = st.number_input("Age", 18, 100)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure", 0, 100)
usage = st.number_input("Usage Frequency", 0, 50)
calls = st.number_input("Support Calls", 0, 20)
delay = st.number_input("Payment Delay", 0, 50)
sub_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
spend = st.number_input("Total Spend", 0)
last = st.number_input("Last Interaction", 0)

if st.button("Predict"):

    df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Tenure': tenure,
        'Usage Frequency': usage,
        'Support Calls': calls,
        'Payment Delay': delay,
        'Subscription Type': sub_type,
        'Contract Length': contract,
        'Total Spend': spend,
        'Last Interaction': last
    }])

    num_cols = ['Age','Tenure','Usage Frequency','Support Calls',
                'Payment Delay','Total Spend','Last Interaction']
    ordinal_cols = ['Subscription Type','Contract Length']
    onehot_cols = ['Gender']

    df[num_cols] = scaler.transform(df[num_cols])
    df[ordinal_cols] = oe.transform(df[ordinal_cols])

    ohe_df = pd.DataFrame(
        ohe.transform(df[onehot_cols]),
        columns=ohe.get_feature_names_out(onehot_cols)
    )

    df = pd.concat([df.drop(columns=onehot_cols), ohe_df], axis=1)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]

    st.write("Prediction:", pred)
    st.write("Probability:", prob)