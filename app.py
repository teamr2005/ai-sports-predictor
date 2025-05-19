import streamlit as st
import random

# 유명 축구 팀 목록
team_names = [
    "Real Madrid", "Barcelona", "Manchester City", "Bayern Munich",
    "Liverpool", "Arsenal", "Paris Saint-Germain", "Chelsea",
    "Juventus", "AC Milan", "Inter Milan", "Atletico Madrid"
]

# 가짜 팀 스탯 (승률, 랭킹, 부상자 수)
team_stats = {
    "Real Madrid": {"win_rate": 0.75, "rank": 1, "injuries": 1},
    "Barcelona": {"win_rate": 0.70, "rank": 2, "injuries": 2},
    "Manchester City": {"win_rate": 0.80, "rank": 1, "injuries": 0},
    "Bayern Munich": {"win_rate": 0.78, "rank": 2, "injuries": 1},
    "Liverpool": {"win_rate": 0.65, "rank": 4, "injuries": 3},
    "Arsenal": {"win_rate": 0.68, "rank": 5, "injuries": 2},
    "Paris Saint-Germain": {"win_rate": 0.72, "rank": 3, "injuries": 1},
    "Chelsea": {"win_rate": 0.55, "rank": 8, "injuries": 4},
    "Juventus": {"win_rate": 0.60, "rank": 6, "injuries": 2},
    "AC Milan": {"win_rate": 0.62, "rank": 7, "injuries": 2},
    "Inter Milan": {"win_rate": 0.66, "rank": 5, "injuries": 1},
    "Atletico Madrid": {"win_rate": 0.64, "rank": 6, "injuries": 2}
}

# 페이지 설정
st.set_page_config(page_title="AI Match Predictor", layout="centered")
st.title("⚽ Real Match AI Predictor (No ML)")
st.markdown("This version uses basic statistics to guess match outcomes.")

# UI
home_team = st.selectbox("Select Home Team", team_names)
away_team = st.selectbox("Select Away Team", team_names)

# 예측
if st.button("Predict Result"):
    if home_team == away_team:
        st.warning("Home and Away team must be different.")
    else:
        # 간단한 점수 계산
        def score(team):
            stat = team_stats[team]
            return stat["win_rate"] * 0.5 + (10 - stat["rank"]) * 0.3 - stat["injuries"] * 0.2

        home_score = score(home_team) + random.uniform(-0.1, 0.1)
        away_score = score(away_team) + random.uniform(-0.1, 0.1)

        # 승패 판단
        if home_score > away_score:
            winner = f"🏠 **{home_team} wins!**"
        elif away_score > home_score:
            winner = f"🛫 **{away_team} wins!**"
        else:
            winner = "⚔️ **Draw**"

        # 점수 예측
        home_goals = max(0, int(round(home_score * 2)))
        away_goals = max(0, int(round(away_score * 2)))

        st.subheader("Prediction Result:")
        st.markdown(winner)
        st.markdown(f"**Predicted Score:** {home_team} {home_goals} - {away_goals} {away_team}")
