import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 예시 팀 및 통계 데이터
teams = ["Man City", "Real Madrid", "Barcelona", "Bayern Munich", "Liverpool"]
team_stats = {
    "Man City": {"ranking": 1, "win_rate": 0.85, "injuries": 0, "form": 0.9},
    "Real Madrid": {"ranking": 2, "win_rate": 0.82, "injuries": 1, "form": 0.85},
    "Barcelona": {"ranking": 3, "win_rate": 0.75, "injuries": 2, "form": 0.8},
    "Bayern Munich": {"ranking": 4, "win_rate": 0.70, "injuries": 0, "form": 0.75},
    "Liverpool": {"ranking": 5, "win_rate": 0.65, "injuries": 1, "form": 0.7},
}

team_logos = {
    "Man City": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "Real Madrid": "https://upload.wikimedia.org/wikipedia/en/5/56/Real_Madrid_CF.svg",
    "Barcelona": "https://upload.wikimedia.org/wikipedia/en/4/47/FC_Barcelona_%28crest%29.svg",
    "Bayern Munich": "https://upload.wikimedia.org/wikipedia/en/1/1f/FC_Bayern_München_logo_%282017%29.svg",
    "Liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg"
}

# 학습용 데이터 생성
data = []
labels = []
for home in teams:
    for away in teams:
        if home != away:
            h = team_stats[home]
            a = team_stats[away]
            row = [
                h["ranking"], h["win_rate"], h["injuries"], h["form"],
                a["ranking"], a["win_rate"], a["injuries"], a["form"]
            ]
            data.append(row)
            labels.append(1 if h["win_rate"] > a["win_rate"] else 0)

X = pd.DataFrame(data, columns=[
    "home_ranking", "home_win_rate", "home_injuries", "home_form",
    "away_ranking", "away_win_rate", "away_injuries", "away_form"
])
y = labels

# 모델 학습
model = RandomForestClassifier()
model.fit(X, y)

# 웹 UI
st.set_page_config(page_title="Soccer Match Predictor", layout="centered")
st.title("🌿 Real Match AI Predictor")
st.markdown("AI predicts the winner based on win rate, ranking, injuries, and recent form!")

home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", teams)

if st.button("Predict Result"):
    if home_team == away_team:
        st.warning("Home and Away teams must be different!")
    else:
        h = team_stats[home_team]
        a = team_stats[away_team]
        input_row = [[
            h["ranking"], h["win_rate"], h["injuries"], h["form"],
            a["ranking"], a["win_rate"], a["injuries"], a["form"]
        ]]
        proba = model.predict_proba(input_row)[0]
        pred = model.predict(input_row)[0]

        st.subheader("Prediction Result:")

        # 팀 로고 표시
        col1, col2 = st.columns(2)
        with col1:
            st.image(team_logos[home_team], width=100, caption=home_team)
        with col2:
            st.image(team_logos[away_team], width=100, caption=away_team)

        if pred == 1:
            st.success(f"🏠 **{home_team} will likely win!**")
        else:
            st.success(f"🛫 **{away_team} will likely win!**")

        st.markdown(f"**{home_team} win probability:** {proba[1]*100:.2f}%")
        st.markdown(f"**{away_team} win probability:** {proba[0]*100:.2f}%")
