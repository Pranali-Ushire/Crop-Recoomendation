from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from crop_recommendation_full_analysis import RANDOM_STATE, add_features, get_models


PROJECT_DIR = Path(r"C:\Users\Arshad Maniyar\OneDrive\Desktop\crop-production\ResearchProject")
DATA_PATH = PROJECT_DIR / "Crop_recommendation.csv"
OUTPUT_DIR = Path.cwd() / "crop_analysis_outputs"
MODEL_DIR = OUTPUT_DIR / "top3_models"

TOP_THREE = [
    "Ensemble_SoftVoting",
    "Ensemble_Stacking",
    "Tree_RandomForest",
]


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = add_features(pd.read_csv(DATA_PATH))
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    X = df.drop(columns=["label"])
    classes = encoder.classes_.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = get_models(X.columns.tolist(), len(classes))
    metadata = {
        "classes": classes,
        "feature_columns": X.columns.tolist(),
        "models": {},
    }

    for model_name in TOP_THREE:
        model = models[model_name]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metadata["models"][model_name] = {
            "display_name": model_name.replace("_", " "),
            "artifact": f"{model_name}.joblib",
            "accuracy": accuracy_score(y_test, predictions),
            "precision_weighted": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
        }
        joblib.dump(model, MODEL_DIR / f"{model_name}.joblib")

    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
