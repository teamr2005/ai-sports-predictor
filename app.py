import streamlit as st
import pandas as pd
import pickle

# 모델 & 팀 목록 불러오기
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("team_names.pkl", "rb") as f:
    team_names = pickle.load(f)

# 웹 UI 구성
st.set_page_config(page_title="AI Match Predictor", layout="centered")
st.title("⚽ Real Match AI Predictor")
st.markdown("Predict real soccer results using machine learning")

# 팀 선택 UI
home_team = st.selectbox("Select Home Team", team_names)
away_team = st.selectbox("Select Away Team", team_names)

# 예측 실행
if st.button("Predict Result"):
    if home_team == away_team:
        st.warning("Home and Away team must be different.")
    else:
        home_id = team_names.index(home_team)
        away_id = team_names.index(away_team)

        input_df = pd.DataFrame([[home_id, away_id]], columns=["home_team_encoded", "away_team_encoded"])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        # 결과 표시
        if pred == 1:
            result = f"🏠 **{home_team} wins**"
        elif pred == -1:
            result = f"✈️ **{away_team} wins**"
        else:
            result = "🤝 **Draw**"

        st.subheader("Prediction Result:")
        st.markdown(result)

        st.markdown(f"**Confidence:** 🟩 Home win: {proba[model.classes_ == 1][0]*100:.2f}%  \n🤝 Draw: {proba[model.classes_ == 0][0]*100:.2f}%  \n🟥 Away win: {proba[model.classes_ == -1][0]*100:.2f}%")
