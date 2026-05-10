import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Investor Retention Prediction")

returns_1yr = st.number_input("1 Year Return")
returns_3yr = st.number_input("3 Year Return")
returns_5yr = st.number_input("5 Year Return")
rating = st.number_input("Rating")
risk_level = st.number_input("Risk Level")

if st.button("Predict"):

    features = np.array([[returns_1yr,
                          returns_3yr,
                          returns_5yr,
                          rating,
                          risk_level]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.warning("Investor likely to churn")
    else:
        st.success("Investor likely to stay")