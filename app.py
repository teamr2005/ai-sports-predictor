import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# --- API SETTINGS ---
API_KEY = "bf063a19180282c3b7fca216afb61509"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# --- HELPER FUNCTIONS ---
def get_team_id(team_name, league_id=39):
    url = f"{BASE_URL}/teams?search={team_name}&league={league_id}&season=2024"
    r = requests.get(url, headers=HEADERS).json()
    return r['response'][0]['team']['id'], r['response'][0]['team']['logo']

def get_recent_form(team_id):
    url = f"{BASE_URL}/fixtures?team={team_id}&last=5"
    r = requests.get(url, headers=HEADERS).json()
    results = r['response']
    wins = sum(1 for game in results if game['teams']['home']['winner'] if game['teams']['home']['id'] == team_id else game['teams']['away']['winner'])
    return wins / 5.0

def get_injuries(team_id):
    url = f"{BASE_URL}/injuries?team={team_id}&season=2024"
    r = requests.get(url, headers=HEADERS).json()
    return len(r['response'])

# --- APP LOGIC ---
st.set_page_config(page_title="AI Soccer Match Predictor", layout="centered")
st.title("⚽ AI Soccer Match Predictor (Live API)")

team1 = st.text_input("Enter Home Team (e.g., Arsenal)")
team2 = st.text_input("Enter Away Team (e.g., Chelsea)")

if team1 and team2 and team1 != team2:
    try:
        id1, logo1 = get_team_id(team1)
        id2, logo2 = get_team_id(team2)

        st.image(logo1, width=100)
        st.image(logo2, width=100)

        inj1 = get_injuries(id1)
        inj2 = get_injuries(id2)

        form1 = get_recent_form(id1)
        form2 = get_recent_form(id2)

        # Features: recent form, injuries
        X = np.array([
            [form1, inj1, form2, inj2]
        ])

        # Train dummy model
        np.random.seed(42)
        dummy_X = np.random.rand(100, 4)
        dummy_y = np.random.randint(0, 5, (100, 2))  # Home and away goals

        model = RandomForestRegressor()
        model.fit(dummy_X, dummy_y)

        pred = model.predict(X)[0]
        home_goals = round(pred[0])
        away_goals = round(pred[1])

        st.subheader("Prediction Result")
        st.write(f"**{team1} {home_goals} - {away_goals} {team2}**")
        st.markdown(f"- **{team1} Injuries**: {inj1}")
        st.markdown(f"- **{team2} Injuries**: {inj2}")
        st.markdown(f"- **{team1} Recent Win Rate**: {form1*100:.1f}%")
        st.markdown(f"- **{team2} Recent Win Rate**: {form2*100:.1f}%")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Enter two different team names to begin.")
