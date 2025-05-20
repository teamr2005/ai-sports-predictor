import streamlit as st
import pandas as pd
import pickle

# Load model and team names
with open("realistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("team_names.pkl", "rb") as f:
    team_names = pickle.load(f)

# Team logos
logos = {
    "Manchester City": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "Arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "Liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
    "Tottenham": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
    "Chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
    "Barcelona": "https://upload.wikimedia.org/wikipedia/en/4/47/FC_Barcelona_%28crest%29.svg",
    "Real Madrid": "https://upload.wikimedia.org/wikipedia/en/5/56/Real_Madrid_CF.svg",
    "Atletico Madrid": "https://upload.wikimedia.org/wikipedia/en/f/f4/Atletico_Madrid_2017_logo.svg",
    "Sevilla": "https://upload.wikimedia.org/wikipedia/en/3/3f/Sevilla_FC_logo.svg",
    "Valencia": "https://upload.wikimedia.org/wikipedia/en/c/ce/Valencia_CF.svg"
}

# Streamlit UI
st.set_page_config(page_title="Real Match AI Predictor", layout="centered")
st.title("🌿 Real Match AI Predictor")
st.markdown("AI predicts the winner based on win rate, ranking, injuries, and recent form!")

home_team = st.selectbox("Select Home Team", team_names)
away_team = st.selectbox("Select Away Team", team_names)

if st.button("Predict Result"):
    if home_team == away_team:
        st.warning("Home and Away teams must be different!")
    else:
        with open("team_names.pkl", "rb") as f:
            names = pickle.load(f)

        home_id = names.index(home_team)
        away_id = names.index(away_team)

        input_df = pd.DataFrame([[home_id, away_id]], columns=["home_id", "away_id"])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        # Show logos and prediction
        col1, col2 = st.columns(2)
        with col1:
            st.image(logos[home_team], width=120)
            st.caption(home_team)
        with col2:
            st.image(logos[away_team], width=120)
            st.caption(away_team)

        st.subheader("Prediction Result:")
        if pred == 1:
            st.success(f"🏠 {home_team} will likely win!")
        else:
            st.success(f"🛫 {away_team} will likely win!")

        st.markdown(f"**{home_team} win probability:** {proba[1]*100:.2f}%")
        st.markdown(f"**{away_team} win probability:** {proba[0]*100:.2f}%")
