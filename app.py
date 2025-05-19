import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 예시 데이터와 팀 이름
teams = ["Man City", "Real Madrid", "Barcelona", "Bayern Munich", "Liverpool"]
team_stats = {
    "Man City": {"ranking": 1, "win_rate": 0.85, "injuries": 0},
    "Real Madrid": {"ranking": 2, "win_rate": 0.82, "injuries": 1},
    "Barcelona": {"ranking": 3, "win_rate": 0.75, "injuries": 2},
    "Bayern Munich": {"ranking": 4, "win_rate": 0.70, "injuries": 0},
    "Liverpool": {"ranking": 5, "win_rate": 0.65, "injuries": 1},
}

# 학습 데이터 구성
data = []
labels = []
for home in teams:
    for away in teams:
        if home != away:
            home_stats = team_stats[home]
            away_stats = team_stats[away]
            row = [
                home_stats["ranking"], home_stats["win_rate"], home_stats["injuries"],
                away_stats["ranking"], away_stats["win_rate"], away_stats["injuries"],
            ]
            data.append(row)
            labels.append(1 if home_stats["win_rate"] > away_stats["win_rate"] else 0)

X = pd.DataFrame(data, columns=[
    "home_ranking", "home_win_rate", "home_injuries",
    "away_ranking", "away_win_rate", "away_injuries"
])
y = labels

# 모델 학습
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="Soccer Match Predictor", layout="centered")
st.title("⚽ Real Match AI Predictor")
st.markdown("AI predicts the winner based on win rate, ranking, and injuries!")

home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", teams)

if st.button("Predict Result"):
    if home_team == away_team:
        st.warning("Home and Away teams must be different!")
    else:
        home = team_stats[home_team]
        away = team_stats[away_team]
        input_row = [[
            home["ranking"], home["win_rate"], home["injuries"],
            away["ranking"], away["win_rate"], away["injuries"]
        ]]
        proba = model.predict_proba(input_row)[0]
        pred = model.predict(input_row)[0]

        st.subheader("Prediction Result:")
        if pred == 1:
            st.success(f"🏠 **{home_team} will likely win!**")
        else:
            st.success(f"🛫 **{away_team} will likely win!**")

        st.markdown(f"**{home_team} win probability:** {proba[1]*100:.2f}%")
        st.markdown(f"**{away_team} win probability:** {proba[0]*100:.2f}%")
