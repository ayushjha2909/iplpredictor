from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from typing import Any

import pandas as pd


MODEL_VERSION = "2.0"
TARGET_COLUMN = "team1_win"

REQUIRED_COLUMNS = [
    "season",
    "city",
    "venue",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "winner",
]

CATEGORICAL_FEATURES = [
    "city",
    "venue",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "home_side",
    "pitch_type",
]

NUMERIC_FEATURES = [
    "season",
    "team1_is_home",
    "team2_is_home",
    "neutral_venue",
    "team1_toss_win",
    "team2_toss_win",
    "team1_overall_win_rate",
    "team2_overall_win_rate",
    "overall_win_rate_diff",
    "team1_recent_win_rate",
    "team2_recent_win_rate",
    "recent_win_rate_diff",
    "team1_previous_match_won",
    "team2_previous_match_won",
    "previous_match_diff",
    "team1_h2h_win_rate",
    "team2_h2h_win_rate",
    "h2h_win_rate_diff",
    "team1_venue_win_rate",
    "team2_venue_win_rate",
    "venue_win_rate_diff",
    "team1_pitch_win_rate",
    "team2_pitch_win_rate",
    "pitch_win_rate_diff",
    "expected_first_innings_score",
    "pace_assist",
    "spin_assist",
    "dew_factor",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

HOME_CITIES = {
    "Chennai Super Kings": "Chennai",
    "Delhi Capitals": "Delhi",
    "Gujarat Titans": "Ahmedabad",
    "Kolkata Knight Riders": "Kolkata",
    "Lucknow Super Giants": "Lucknow",
    "Mumbai Indians": "Mumbai",
    "Punjab Kings": "Mohali",
    "Rajasthan Royals": "Jaipur",
    "Royal Challengers Bengaluru": "Bengaluru",
    "Sunrisers Hyderabad": "Hyderabad",
}

PITCH_PROFILES = {
    "Arun Jaitley Stadium": {
        "pitch_type": "batting",
        "expected_first_innings_score": 173,
        "pace_assist": 0.45,
        "spin_assist": 0.55,
        "dew_factor": 0.55,
    },
    "BRSABV Ekana Cricket Stadium": {
        "pitch_type": "slow",
        "expected_first_innings_score": 158,
        "pace_assist": 0.40,
        "spin_assist": 0.70,
        "dew_factor": 0.35,
    },
    "Barsapara Cricket Stadium": {
        "pitch_type": "batting",
        "expected_first_innings_score": 180,
        "pace_assist": 0.45,
        "spin_assist": 0.45,
        "dew_factor": 0.55,
    },
    "Brabourne Stadium": {
        "pitch_type": "batting",
        "expected_first_innings_score": 182,
        "pace_assist": 0.50,
        "spin_assist": 0.40,
        "dew_factor": 0.65,
    },
    "DY Patil Stadium": {
        "pitch_type": "balanced",
        "expected_first_innings_score": 170,
        "pace_assist": 0.55,
        "spin_assist": 0.45,
        "dew_factor": 0.60,
    },
    "Dubai International Cricket Stadium": {
        "pitch_type": "balanced",
        "expected_first_innings_score": 166,
        "pace_assist": 0.50,
        "spin_assist": 0.55,
        "dew_factor": 0.50,
    },
    "Eden Gardens": {
        "pitch_type": "batting",
        "expected_first_innings_score": 176,
        "pace_assist": 0.50,
        "spin_assist": 0.45,
        "dew_factor": 0.60,
    },
    "IS Bindra Stadium": {
        "pitch_type": "pace friendly",
        "expected_first_innings_score": 171,
        "pace_assist": 0.70,
        "spin_assist": 0.35,
        "dew_factor": 0.55,
    },
    "M. A. Chidambaram Stadium": {
        "pitch_type": "spin friendly",
        "expected_first_innings_score": 164,
        "pace_assist": 0.35,
        "spin_assist": 0.75,
        "dew_factor": 0.40,
    },
    "M. Chinnaswamy Stadium": {
        "pitch_type": "batting",
        "expected_first_innings_score": 186,
        "pace_assist": 0.45,
        "spin_assist": 0.35,
        "dew_factor": 0.70,
    },
    "MCA Stadium": {
        "pitch_type": "balanced",
        "expected_first_innings_score": 171,
        "pace_assist": 0.55,
        "spin_assist": 0.45,
        "dew_factor": 0.45,
    },
    "Narendra Modi Stadium": {
        "pitch_type": "balanced",
        "expected_first_innings_score": 172,
        "pace_assist": 0.60,
        "spin_assist": 0.45,
        "dew_factor": 0.50,
    },
    "Rajiv Gandhi International Stadium": {
        "pitch_type": "batting",
        "expected_first_innings_score": 179,
        "pace_assist": 0.45,
        "spin_assist": 0.45,
        "dew_factor": 0.60,
    },
    "Sawai Mansingh Stadium": {
        "pitch_type": "slow",
        "expected_first_innings_score": 162,
        "pace_assist": 0.45,
        "spin_assist": 0.65,
        "dew_factor": 0.35,
    },
    "Sharjah Cricket Stadium": {
        "pitch_type": "batting",
        "expected_first_innings_score": 181,
        "pace_assist": 0.40,
        "spin_assist": 0.50,
        "dew_factor": 0.45,
    },
    "Sheikh Zayed Stadium": {
        "pitch_type": "balanced",
        "expected_first_innings_score": 163,
        "pace_assist": 0.60,
        "spin_assist": 0.50,
        "dew_factor": 0.45,
    },
    "Wankhede Stadium": {
        "pitch_type": "batting",
        "expected_first_innings_score": 184,
        "pace_assist": 0.60,
        "spin_assist": 0.35,
        "dew_factor": 0.75,
    },
}

DEFAULT_PITCH_PROFILE = {
    "pitch_type": "balanced",
    "expected_first_innings_score": 170,
    "pace_assist": 0.50,
    "spin_assist": 0.50,
    "dew_factor": 0.50,
}

PITCH_TYPES = ["balanced", "batting", "spin friendly", "pace friendly", "slow"]


def normalize_team_name(team: Any) -> Any:
    if pd.isna(team):
        return team
    cleaned = str(team).strip()
    return TEAM_ALIASES.get(cleaned, cleaned)


def get_pitch_profile(venue: str) -> dict[str, Any]:
    return deepcopy(PITCH_PROFILES.get(str(venue).strip(), DEFAULT_PITCH_PROFILE))


def normalize_matches(data: pd.DataFrame, require_winner: bool = True) -> pd.DataFrame:
    normalized = data.copy()
    required_columns = REQUIRED_COLUMNS if require_winner else REQUIRED_COLUMNS[:-1]
    missing_columns = [column for column in required_columns if column not in normalized.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    normalized["_source_order"] = range(len(normalized))

    for column in ["team1", "team2", "toss_winner", "winner"]:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(normalize_team_name)

    normalized["city"] = normalized["city"].fillna("Unknown").astype(str).str.strip()
    normalized["venue"] = normalized["venue"].fillna("Unknown").astype(str).str.strip()
    normalized["toss_decision"] = (
        normalized["toss_decision"]
        .fillna("field")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"bowl": "field", "bowling": "field", "batting": "bat"})
    )
    normalized["season"] = pd.to_numeric(normalized["season"], errors="coerce").fillna(0).astype(int)

    pitch_profiles = normalized["venue"].map(get_pitch_profile)
    for column in ["pitch_type", "expected_first_innings_score", "pace_assist", "spin_assist", "dew_factor"]:
        default_values = pitch_profiles.map(lambda profile: profile[column])
        if column in normalized.columns:
            normalized[column] = normalized[column].where(normalized[column].notna(), default_values)
        else:
            normalized[column] = default_values

    normalized["pitch_type"] = normalized["pitch_type"].astype(str).str.strip().str.lower()
    for column in ["expected_first_innings_score", "pace_assist", "spin_assist", "dew_factor"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(
            DEFAULT_PITCH_PROFILE[column]
        )

    return normalized.sort_values(["season", "_source_order"]).reset_index(drop=True)


def new_history_state() -> dict[str, Any]:
    return {
        "team": defaultdict(lambda: {"matches": 0, "wins": 0, "recent": deque(maxlen=5)}),
        "h2h": defaultdict(lambda: {"matches": 0, "wins": defaultdict(int)}),
        "venue": defaultdict(lambda: {"matches": 0, "wins": 0}),
        "pitch": defaultdict(lambda: {"matches": 0, "wins": 0}),
    }


def smoothed_win_rate(wins: float, matches: float, prior: float = 0.5, strength: float = 2.0) -> float:
    return float((wins + prior * strength) / (matches + strength))


def recent_win_rate(recent_results: deque[int]) -> float:
    if not recent_results:
        return 0.5
    return float(sum(recent_results) / len(recent_results))


def previous_match_won(recent_results: deque[int]) -> float:
    if not recent_results:
        return 0.5
    return float(recent_results[-1])


def detect_home_side(team1: str, team2: str, city: str) -> str:
    team1_home = HOME_CITIES.get(team1) == city
    team2_home = HOME_CITIES.get(team2) == city
    if team1_home and not team2_home:
        return "team1"
    if team2_home and not team1_home:
        return "team2"
    if team1_home and team2_home:
        return "shared"
    return "neutral"


def row_to_features(row: pd.Series, state: dict[str, Any]) -> dict[str, Any]:
    team1 = row["team1"]
    team2 = row["team2"]
    venue = row["venue"]
    city = row["city"]
    pitch_type = row["pitch_type"]

    team1_stats = state["team"][team1]
    team2_stats = state["team"][team2]
    team1_recent = recent_win_rate(team1_stats["recent"])
    team2_recent = recent_win_rate(team2_stats["recent"])
    team1_previous = previous_match_won(team1_stats["recent"])
    team2_previous = previous_match_won(team2_stats["recent"])

    h2h_key = tuple(sorted([team1, team2]))
    h2h_stats = state["h2h"][h2h_key]
    team1_h2h = smoothed_win_rate(h2h_stats["wins"][team1], h2h_stats["matches"])
    team2_h2h = smoothed_win_rate(h2h_stats["wins"][team2], h2h_stats["matches"])

    team1_venue_stats = state["venue"][(team1, venue)]
    team2_venue_stats = state["venue"][(team2, venue)]
    team1_venue = smoothed_win_rate(team1_venue_stats["wins"], team1_venue_stats["matches"])
    team2_venue = smoothed_win_rate(team2_venue_stats["wins"], team2_venue_stats["matches"])

    team1_pitch_stats = state["pitch"][(team1, pitch_type)]
    team2_pitch_stats = state["pitch"][(team2, pitch_type)]
    team1_pitch = smoothed_win_rate(team1_pitch_stats["wins"], team1_pitch_stats["matches"])
    team2_pitch = smoothed_win_rate(team2_pitch_stats["wins"], team2_pitch_stats["matches"])

    team1_overall = smoothed_win_rate(team1_stats["wins"], team1_stats["matches"])
    team2_overall = smoothed_win_rate(team2_stats["wins"], team2_stats["matches"])

    home_side = detect_home_side(team1, team2, city)
    team1_is_home = int(home_side in {"team1", "shared"})
    team2_is_home = int(home_side in {"team2", "shared"})

    features = {
        "season": int(row["season"]),
        "city": city,
        "venue": venue,
        "team1": team1,
        "team2": team2,
        "toss_winner": row["toss_winner"],
        "toss_decision": row["toss_decision"],
        "home_side": home_side,
        "pitch_type": pitch_type,
        "team1_is_home": team1_is_home,
        "team2_is_home": team2_is_home,
        "neutral_venue": int(home_side == "neutral"),
        "team1_toss_win": int(row["toss_winner"] == team1),
        "team2_toss_win": int(row["toss_winner"] == team2),
        "team1_overall_win_rate": team1_overall,
        "team2_overall_win_rate": team2_overall,
        "overall_win_rate_diff": team1_overall - team2_overall,
        "team1_recent_win_rate": team1_recent,
        "team2_recent_win_rate": team2_recent,
        "recent_win_rate_diff": team1_recent - team2_recent,
        "team1_previous_match_won": team1_previous,
        "team2_previous_match_won": team2_previous,
        "previous_match_diff": team1_previous - team2_previous,
        "team1_h2h_win_rate": team1_h2h,
        "team2_h2h_win_rate": team2_h2h,
        "h2h_win_rate_diff": team1_h2h - team2_h2h,
        "team1_venue_win_rate": team1_venue,
        "team2_venue_win_rate": team2_venue,
        "venue_win_rate_diff": team1_venue - team2_venue,
        "team1_pitch_win_rate": team1_pitch,
        "team2_pitch_win_rate": team2_pitch,
        "pitch_win_rate_diff": team1_pitch - team2_pitch,
        "expected_first_innings_score": float(row["expected_first_innings_score"]),
        "pace_assist": float(row["pace_assist"]),
        "spin_assist": float(row["spin_assist"]),
        "dew_factor": float(row["dew_factor"]),
    }
    return features


def update_history_state(row: pd.Series, state: dict[str, Any]) -> None:
    winner = row.get("winner")
    team1 = row["team1"]
    team2 = row["team2"]
    if winner not in {team1, team2}:
        return

    venue = row["venue"]
    pitch_type = row["pitch_type"]
    h2h_key = tuple(sorted([team1, team2]))

    for team in [team1, team2]:
        won = int(winner == team)
        team_stats = state["team"][team]
        team_stats["matches"] += 1
        team_stats["wins"] += won
        team_stats["recent"].append(won)

        venue_stats = state["venue"][(team, venue)]
        venue_stats["matches"] += 1
        venue_stats["wins"] += won

        pitch_stats = state["pitch"][(team, pitch_type)]
        pitch_stats["matches"] += 1
        pitch_stats["wins"] += won

    state["h2h"][h2h_key]["matches"] += 1
    state["h2h"][h2h_key]["wins"][winner] += 1


def build_training_frame(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = normalize_matches(raw_data, require_winner=True)
    state = new_history_state()
    rows = []

    for _, row in matches.iterrows():
        if row["winner"] not in {row["team1"], row["team2"]}:
            update_history_state(row, state)
            continue
        features = row_to_features(row, state)
        features[TARGET_COLUMN] = int(row["winner"] == row["team1"])
        features["winner"] = row["winner"]
        rows.append(features)
        update_history_state(row, state)

    if not rows:
        raise ValueError("Dataset has no usable completed matches.")

    return pd.DataFrame(rows), matches


def build_prediction_frame(history: pd.DataFrame, match: dict[str, Any]) -> pd.DataFrame:
    history_matches = normalize_matches(history, require_winner=True)
    match_frame = normalize_matches(pd.DataFrame([match]), require_winner=False)
    match_row = match_frame.iloc[0]
    state = new_history_state()

    prior_history = history_matches[history_matches["season"] < int(match_row["season"])]
    if prior_history.empty:
        prior_history = history_matches

    for _, history_row in prior_history.iterrows():
        update_history_state(history_row, state)

    features = row_to_features(match_row, state)
    return pd.DataFrame([features], columns=MODEL_FEATURES)
