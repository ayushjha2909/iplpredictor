from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.predict import DEFAULT_MODEL_PATH, predict_winner_probability
from src.features import MODEL_VERSION, PITCH_TYPES, get_pitch_profile


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "sample_ipl_matches.csv"


def train_current_model() -> None:
    from src.train_model import train_model

    train_model(DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH)


st.set_page_config(page_title="IPL Winner Predictor", layout="centered")

TEAM_OPTIONS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

VENUE_BY_CITY = {
    "Ahmedabad": ["Narendra Modi Stadium"],
    "Bengaluru": ["M. Chinnaswamy Stadium"],
    "Chennai": ["M. A. Chidambaram Stadium"],
    "Delhi": ["Arun Jaitley Stadium"],
    "Hyderabad": ["Rajiv Gandhi International Stadium"],
    "Jaipur": ["Sawai Mansingh Stadium"],
    "Kolkata": ["Eden Gardens"],
    "Lucknow": ["BRSABV Ekana Cricket Stadium"],
    "Mohali": ["IS Bindra Stadium"],
    "Mumbai": ["Wankhede Stadium", "Brabourne Stadium", "DY Patil Stadium"],
    "Pune": ["MCA Stadium"],
}


@st.cache_resource
def ensure_model(model_path: Path) -> dict:
    if not model_path.exists():
        train_current_model()
    model_bundle = joblib.load(model_path)
    if model_bundle.get("model_version") != MODEL_VERSION:
        train_current_model()
        model_bundle = joblib.load(model_path)
    return model_bundle


st.title("IPL Match Winner Probability")
st.caption("Random Forest model for match-level winner probability prediction")

model_bundle = ensure_model(DEFAULT_MODEL_PATH)

with st.sidebar:
    st.header("Model")
    st.metric("Training rows", int(model_bundle.get("training_rows", 0)))
    st.metric("Saved test accuracy", f"{model_bundle.get('accuracy', 0):.1%}")
    with st.expander("Top factors"):
        top_features = model_bundle.get("top_features")
        if top_features is not None:
            st.dataframe(top_features, hide_index=True, use_container_width=True)
    if st.button("Retrain model"):
        train_current_model()
        st.cache_resource.clear()
        st.rerun()

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", TEAM_OPTIONS, index=5)
with col2:
    team2_options = [team for team in TEAM_OPTIONS if team != team1]
    team2 = st.selectbox("Team 2", team2_options, index=0)

city = st.selectbox("City", list(VENUE_BY_CITY.keys()), index=list(VENUE_BY_CITY.keys()).index("Mumbai"))
venue = st.selectbox("Venue", VENUE_BY_CITY[city])
pitch_profile = get_pitch_profile(venue)
pitch_type = st.selectbox(
    "Pitch report",
    PITCH_TYPES,
    index=PITCH_TYPES.index(pitch_profile["pitch_type"]),
)
toss_winner = st.selectbox("Toss winner", [team1, team2])
toss_decision = st.radio("Toss decision", ["bat", "field"], horizontal=True)
season = st.number_input("Season", min_value=2008, max_value=2030, value=2024, step=1)

score_col, pace_col = st.columns(2)
with score_col:
    expected_score = st.slider(
        "Expected first innings score",
        min_value=120,
        max_value=240,
        value=int(pitch_profile["expected_first_innings_score"]),
        step=1,
    )
with pace_col:
    dew_factor = st.slider(
        "Dew factor",
        min_value=0.0,
        max_value=1.0,
        value=float(pitch_profile["dew_factor"]),
        step=0.05,
    )

assist_col1, assist_col2 = st.columns(2)
with assist_col1:
    pace_assist = st.slider(
        "Pace assist",
        min_value=0.0,
        max_value=1.0,
        value=float(pitch_profile["pace_assist"]),
        step=0.05,
    )
with assist_col2:
    spin_assist = st.slider(
        "Spin assist",
        min_value=0.0,
        max_value=1.0,
        value=float(pitch_profile["spin_assist"]),
        step=0.05,
    )

if st.button("Predict winner probability", type="primary"):
    probabilities = predict_winner_probability(
        team1=team1,
        team2=team2,
        city=city,
        venue=venue,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
        season=int(season),
        pitch_type=pitch_type,
        expected_first_innings_score=float(expected_score),
        pace_assist=float(pace_assist),
        spin_assist=float(spin_assist),
        dew_factor=float(dew_factor),
    )

    predicted = probabilities.iloc[0]
    st.subheader(f"Predicted winner: {predicted['team']}")
    st.metric("Win probability", f"{predicted['win_probability']:.1%}")

    chart_data = probabilities.set_index("team")["win_probability"]
    st.bar_chart(chart_data)

    st.dataframe(
        probabilities.assign(win_probability=lambda frame: frame["win_probability"].map(lambda value: f"{value:.1%}")),
        hide_index=True,
        use_container_width=True,
    )

with st.expander("Dataset preview"):
    data = pd.read_csv(DEFAULT_DATA_PATH)
    st.dataframe(data.tail(10), use_container_width=True, hide_index=True)
