import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("soccer_model.pkl")

st.set_page_config(page_title="AI Soccer Match Predictor", layout="centered")
st.title("⚽ AI Soccer Match Predictor")

st.markdown("Predict the chance of home team winning based on simple team stats.")

home_id = st.number_input("Enter Home Team ID", value=9991)
away_id = st.number_input("Enter Away Team ID", value=9998)

if st.button("Predict Result"):
    input_data = pd.DataFrame([[home_id, away_id]], columns=["home_team_api_id", "away_team_api_id"])
    prediction = model.predict_proba(input_data)[0]
    st.markdown(f"**Home Win Probability:** {prediction[1]*100:.2f}%")
    st.markdown(f"**Away Win Probability:** {prediction[0]*100:.2f}%")
