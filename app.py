import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 샘플 학습용 데이터
data = pd.DataFrame({
    "home_team_id": [1, 2, 3, 4, 5, 6],
    "away_team_id": [2, 3, 1, 5, 6, 4],
    "home_win":     [1, 0, 1, 0, 1, 0]
})

X = data[["home_team_id", "away_team_id"]]
y = data["home_win"]

# 모델 학습
model = RandomForestClassifier()
model.fit(X, y)

# 웹 UI
st.title("⚽ AI Soccer Match Predictor")
st.markdown("Simple demo using RandomForest on example data")

home_id = st.number_input("Enter Home Team ID", value=1)
away_id = st.number_input("Enter Away Team ID", value=2)

if st.button("Predict Result"):
    input_data = pd.DataFrame([[home_id, away_id]], columns=["home_team_id", "away_team_id"])
    proba = model.predict_proba(input_data)[0]
    st.markdown(f"**Home Win Probability:** {proba[1]*100:.2f}%")
    st.markdown(f"**Away Win Probability:** {proba[0]*100:.2f}%")
