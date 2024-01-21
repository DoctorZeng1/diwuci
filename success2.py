import streamlit as st
import pandas as pd
import joblib
import shap
import shap.plots
import matplotlib.pyplot as plt

# Title
st.header("Streamlit Machine Learning App")

# Input bar 1
volume = st.number_input("Enter Stone volume")

# Input bar 2
pressures = st.number_input("Enter pressures")

# Input bar 3
time = st.number_input("Enter time")

# Dropdown input4
WBC = st.selectbox("Select WBC Inspection Results", ("positive", "negative"))

# Dropdown input5
Nitrite = st.selectbox("Select Nitrite Inspection Results", ("Positive", "Negative"))

# Input bar 6
scores = st.number_input("Enter scores")

# If button is pressed
if st.button("Predict"):
    # Unpickle classifier
    classifier2 = joblib.load("classifier2.pkl")

    # Store inputs into dataframe
    X = pd.DataFrame([[volume, pressures, time, WBC, Nitrite, scores]],
                     columns=["volume", "pressures", "time", "WBC", "Nitrite", "scores"])
    X = X.replace(["positive", "negative"], [1, 0])
    X = X.replace(["Positive", "Negative"], [1, 0])

    # Get prediction probability
    prediction_proba = classifier2.predict_proba(X)

    # Output prediction probability
    st.text(f"The probability of positive outcome is {prediction_proba[0, 1]}")

    # SHAP explanation
    explainer = shap.TreeExplainer(classifier2)
    shap_values = explainer.shap_values(X)

    # Plot SHAP decision plot
    expected_value = explainer.expected_value
    fig, ax = plt.subplots()
    shap.decision_plot(expected_value, shap_values, X)
    st.pyplot(fig)

