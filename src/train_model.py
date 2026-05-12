from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from src.features import (
        CATEGORICAL_FEATURES,
        MODEL_VERSION,
        MODEL_FEATURES,
        NUMERIC_FEATURES,
        TARGET_COLUMN,
        build_training_frame,
    )
except ModuleNotFoundError:
    from features import (
        CATEGORICAL_FEATURES,
        MODEL_VERSION,
        MODEL_FEATURES,
        NUMERIC_FEATURES,
        TARGET_COLUMN,
        build_training_frame,
    )


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "sample_ipl_matches.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "ipl_winner_random_forest.joblib"


def load_dataset(data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    data = data.dropna(subset=["winner"])
    if data.empty:
        raise ValueError("Dataset has no usable rows after dropping records with missing winners.")

    return data


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_FEATURES,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def get_feature_importance(pipeline: Pipeline, top_n: int = 10) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def train_model(data_path: Path, model_path: Path, test_size: float = 0.25) -> dict[str, float]:
    raw_data = load_dataset(data_path)
    data, history_data = build_training_frame(raw_data)
    x = data[MODEL_FEATURES]
    y = data[TARGET_COLUMN]

    split_index = max(1, min(len(data) - 1, int(len(data) * (1 - test_size))))
    x_train = x.iloc[:split_index]
    y_train = y.iloc[:split_index]
    x_test = x.iloc[split_index:]
    y_test = y.iloc[split_index:]

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "model_version": MODEL_VERSION,
            "feature_columns": MODEL_FEATURES,
            "target_column": TARGET_COLUMN,
            "history_data": history_data,
            "teams": sorted(set(history_data["team1"]).union(set(history_data["team2"]))),
            "training_rows": len(data),
            "accuracy": accuracy,
            "top_features": get_feature_importance(pipeline, top_n=12),
        },
        model_path,
    )

    print(f"Training rows: {len(x_train)}")
    print(f"Testing rows: {len(x_test)}")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["team2_win", "team1_win"],
            zero_division=0,
        )
    )
    print("\nTop model factors:")
    print(get_feature_importance(pipeline, top_n=12).to_string(index=False))
    print(f"\nSaved model to: {model_path}")

    return {"accuracy": accuracy, "training_rows": float(len(x_train)), "testing_rows": float(len(x_test))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an IPL winner probability Random Forest model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the IPL match CSV file.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path where the trained model is saved.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraction of rows used for testing.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.data, args.model, args.test_size)
