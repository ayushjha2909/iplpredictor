from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

try:
    from src.features import DEFAULT_PITCH_PROFILE, MODEL_VERSION, MODEL_FEATURES, PITCH_TYPES, build_prediction_frame
except ModuleNotFoundError:
    from features import DEFAULT_PITCH_PROFILE, MODEL_VERSION, MODEL_FEATURES, PITCH_TYPES, build_prediction_frame


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "ipl_winner_random_forest.joblib"


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train it first with: python src\\train_model.py"
        )
    model_bundle = joblib.load(model_path)
    if model_bundle.get("model_version") != MODEL_VERSION:
        raise ValueError("Model file is from an older project version. Retrain it with: python src\\train_model.py")
    return model_bundle


def predict_winner_probability(
    team1: str,
    team2: str,
    city: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    season: int,
    pitch_type: str | None = None,
    expected_first_innings_score: float | None = None,
    pace_assist: float | None = None,
    spin_assist: float | None = None,
    dew_factor: float | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    include_all_teams: bool = False,
) -> pd.DataFrame:
    model_bundle = load_model(model_path)
    pipeline = model_bundle["pipeline"]

    match = {
        "season": season,
        "city": city,
        "venue": venue,
        "team1": team1,
        "team2": team2,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
    }
    if pitch_type is not None:
        match["pitch_type"] = pitch_type
    if expected_first_innings_score is not None:
        match["expected_first_innings_score"] = expected_first_innings_score
    if pace_assist is not None:
        match["pace_assist"] = pace_assist
    if spin_assist is not None:
        match["spin_assist"] = spin_assist
    if dew_factor is not None:
        match["dew_factor"] = dew_factor
    features = build_prediction_frame(model_bundle["history_data"], match)

    probabilities = pipeline.predict_proba(features[MODEL_FEATURES])[0]
    classes = pipeline.classes_
    probability_by_class = dict(zip(classes, probabilities))
    team1_probability = float(probability_by_class.get(1, 0.0))
    team2_probability = float(probability_by_class.get(0, 0.0))

    result = pd.DataFrame(
        [
            {"team": team1, "win_probability": team1_probability},
            {"team": team2, "win_probability": team2_probability},
        ]
    ).sort_values("win_probability", ascending=False)

    probability_sum = result["win_probability"].sum()
    if probability_sum > 0:
        result["win_probability"] = result["win_probability"] / probability_sum

    if include_all_teams:
        factor_rows = features.iloc[0].to_dict()
        result.attrs["features"] = factor_rows

    return result.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict IPL match winner probability.")
    parser.add_argument("--team1", required=True)
    parser.add_argument("--team2", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--venue", required=True)
    parser.add_argument("--toss-winner", required=True)
    parser.add_argument("--toss-decision", required=True, choices=["bat", "field"])
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--pitch-type", choices=PITCH_TYPES, help="Pitch report category. Defaults to venue profile.")
    parser.add_argument("--expected-score", type=float, help="Expected first innings score. Defaults to venue profile.")
    parser.add_argument("--pace-assist", type=float, help="Pace assistance from 0 to 1. Defaults to venue profile.")
    parser.add_argument("--spin-assist", type=float, help="Spin assistance from 0 to 1. Defaults to venue profile.")
    parser.add_argument("--dew-factor", type=float, help="Dew impact from 0 to 1. Defaults to venue profile.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--show-factors", action="store_true", help="Show the engineered feature values used by the model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prediction = predict_winner_probability(
        team1=args.team1,
        team2=args.team2,
        city=args.city,
        venue=args.venue,
        toss_winner=args.toss_winner,
        toss_decision=args.toss_decision,
        season=args.season,
        pitch_type=args.pitch_type,
        expected_first_innings_score=args.expected_score,
        pace_assist=args.pace_assist,
        spin_assist=args.spin_assist,
        dew_factor=args.dew_factor,
        model_path=args.model,
        include_all_teams=args.show_factors,
    )

    top_team = prediction.iloc[0]
    print(f"Predicted winner: {top_team['team']} ({top_team['win_probability']:.1%})")
    print("\nAll probabilities:")
    for _, row in prediction.iterrows():
        print(f"- {row['team']}: {row['win_probability']:.1%}")

    if args.show_factors:
        print("\nEngineered factors:")
        for key, value in prediction.attrs["features"].items():
            print(f"- {key}: {value}")
