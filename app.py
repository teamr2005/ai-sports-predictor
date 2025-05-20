import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

# API 설정
API_KEY = "bf063a19180282c3b7fca216afb61509"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "v3.football.api-sports.io"
}

# 팀 리스트와 ID 매핑
teams = {
    "Man City": 50,
    "Real Madrid": 541,
    "Barcelona": 529,
    "Bayern Munich": 157,
    "Liverpool": 40
}

# 부상자 수 불러오기
def get_injuries(team_id):
    url = f"{BASE_URL}/injuries?team={team_id}&season=2023"
    r = requests.get(url, headers=HEADERS).json()
    return len(r['response'])

# 최근 폼 점수 불러오기 (마지막 5경기 중 승리 비율)
def get_recent_form(team_id):
    url = f"{BASE_URL}/fixtures?team={team_id}&last=5"
    r = requests.get(url, headers=HEADERS).json()
    results = r['response']
    wins = 0
    for game in results:
        if game['teams']['home']['id'] == team_id:
            if game['teams']['home']['winner']:
                wins += 1
        else:
            if game['teams']['away']['winner']:
                wins += 1
    return wins / 5.0

# 승률과 랭킹 임시 하드코딩 (API에서 자동화 가능)
team_stats = {
    "Man City": {"ranking": 1, "win_rate": 0.85},
    "Real Madrid": {"ranking": 2, "win_rate": 0.82},
    "Barcelona": {"ranking": 3, "win_rate": 0.75},
    "Bayern Munich": {"ranking": 4, "win_rate": 0.70},
    "Liverpool": {"ranking": 5, "win_rate": 0.65},
}

# Streamlit 설정
st.set_page_config(page_title="Soccer Match Predictor", layout="centered")
st.title("🌿 Real Match AI Predictor")
st.markdown("AI predicts the winner based on win rate, ranking, injuries, and recent form!")

home_team = st.selectbox("Select Home Team", list(teams.keys()))
away_team = st.selectbox("Select Away Team", list(teams.keys()))

if st.button("Predict Result"):
    if home_team == away_team:
        st.warning("Home and Away teams must be different!")
    else:
        # 부상 및 최근폼 불러오기
        for team in [home_team, away_team]:
            team_stats[team]["injuries"] = get_injuries(teams[team])
            team_stats[team]["form"] = get_recent_form(teams[team])

        # 모델 학습 데이터 구성
        data = []
        labels = []
        for h in teams.keys():
            for a in teams.keys():
                if h != a:
                    h_s, a_s = team_stats[h], team_stats[a]
                    row = [
                        h_s["ranking"], h_s["win_rate"], h_s["injuries"], h_s["form"],
                        a_s["ranking"], a_s["win_rate"], a_s["injuries"], a_s["form"]
                    ]
                    data.append(row)
                    labels.append(1 if h_s["win_rate"] > a_s["win_rate"] else 0)

        X = pd.DataFrame(data, columns=[
            "home_rank", "home_winrate", "home_injuries", "home_form",
            "away_rank", "away_winrate", "away_injuries", "away_form"
        ])
        y = labels

        model = RandomForestClassifier()
        model.fit(X, y)

        # 예측용 입력값 생성
        h_s, a_s = team_stats[home_team], team_stats[away_team]
        input_row = [[
            h_s["ranking"], h_s["win_rate"], h_s["injuries"], h_s["form"],
            a_s["ranking"], a_s["win_rate"], a_s["injuries"], a_s["form"]
        ]]

        pred = model.predict(input_row)[0]
        proba = model.predict_proba(input_row)[0]

        st.subheader("Prediction Result:")
        if pred == 1:
            st.success(f"🏠 **{home_team} will likely win!**")
        else:
            st.success(f"🛫 **{away_team} will likely win!**")

        st.markdown(f"**{home_team} win probability:** {proba[1]*100:.2f}%")
        st.markdown(f"**{away_team} win probability:** {proba[0]*100:.2f}%")
