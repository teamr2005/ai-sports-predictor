import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 데이터 (예시 - 유명 축구팀)
team_names = [
    "Real Madrid", "Barcelona", "Manchester City", "Liverpool",
    "Bayern Munich", "Arsenal", "PSG", "Juventus"
]

# 팀별 임의 승률, 랭킹, 부상자 수 (예시값)
team_stats = {
    "Real Madrid":       {"win_rate": 0.8, "ranking": 1, "injuries": 1},
    "Barcelona":         {"win_rate": 0.7, "ranking": 2, "injuries": 2},
    "Manchester City":   {"win_rate": 0.75, "ranking": 1, "injuries": 1},
    "Liverpool":         {"win_rate": 0.65, "ranking": 3, "injuries": 3},
    "Bayern Munich":     {"win_rate": 0.78, "ranking": 1, "injuries": 1},
    "Arsenal":           {"win_rate": 0.6, "ranking": 4, "injuries": 2},
    "PSG":               {"win_rate": 0.7, "ranking": 2, "injuries": 2},
    "Juventus":          {"win_rate": 0.5, "ranking": 5, "injuries": 4}
}

# 훈련용 더미 데이터 생성
X = []
y = []

for home in team_names:
    for away in team_names:
        if home == away:
            continue
        h = team_stats[home]
        a = team_stats[away]
        features = [
            h["win_rate"], h["ranking"], h["injuries"],
            a["win_rate"], a["ranking"], a["injuries"]
        ]
        # 단순히 승률 높은 팀이 이기는 것으로 예측 (예시)
        label = 1 if h["win_rate"] > a["win_rate"] else -1 if h["win_rate"] < a["win_rate"] else 0
        X.append(features)
        y.append(label)

model = RandomForestClassifier()
model.fit(X, y)

# UI
st.set_page_config(page_title="Soccer Predictor", layout="centered")
st.title("⚽ AI Match Predictor")
st.markdown("Predict outcome using AI based on stats (win rate, ranking, injuries)")

home_team = st.selectbox("🏠 Home Team", team_names)
away_team = st.selectbox("🛫 Away Team", team_names)

if st.button("Predict Result"):
    if home_team == away_team:
        st.warning("Teams must be different.")
    else:
        h = team_stats[home_team]
        a = team_stats[away_team]
        input_data = [[
            h["win_rate"], h["ranking"], h["injuries"],
            a["win_rate"], a["ranking"], a["injuries"]
        ]]
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        if pred == 1:
            st.success(f"🏠 **{home_team} wins**")
        elif pred == -1:
            st.success(f"🛫 **{away_team} wins**")
        else:
            st.info("⚔️ **Draw**")

        st.markdown(f"**Home Win Probability:** {proba[model.classes_ == 1][0]*100:.2f}%")
        st.markdown(f"**Away Win Probability:** {proba[model.classes_ == -1][0]*100:.2f}%")
