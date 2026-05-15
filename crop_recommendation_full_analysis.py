from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")


PROJECT_DIR = Path(r"C:\Users\Arshad Maniyar\OneDrive\Desktop\crop-production\ResearchProject")
DATA_PATH = PROJECT_DIR / "Crop_recommendation.csv"
OUTPUT_DIR = Path.cwd() / "crop_analysis_outputs"
RANDOM_STATE = 42


def make_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "confusion_matrices").mkdir(exist_ok=True)


def load_and_validate_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    expected_columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]

    validation = {
        "source_file": str(DATA_PATH),
        "rows_read": int(len(df)),
        "columns_read": list(df.columns),
        "expected_columns_match": list(df.columns) == expected_columns,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {column: int(count) for column, count in df.isna().sum().items()},
        "crop_classes": sorted(df["label"].unique().tolist()),
        "rows_per_crop": {k: int(v) for k, v in df["label"].value_counts().sort_index().items()},
    }
    (OUTPUT_DIR / "dataset_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    df.describe(include="all").transpose().to_csv(OUTPUT_DIR / "dataset_summary.csv")
    df.groupby("label").agg(
        rows=("label", "size"),
        N_min=("N", "min"),
        N_median=("N", "median"),
        N_max=("N", "max"),
        P_min=("P", "min"),
        P_median=("P", "median"),
        P_max=("P", "max"),
        K_min=("K", "min"),
        K_median=("K", "median"),
        K_max=("K", "max"),
        temperature_median=("temperature", "median"),
        humidity_median=("humidity", "median"),
        ph_median=("ph", "median"),
        rainfall_median=("rainfall", "median"),
    ).round(3).to_csv(OUTPUT_DIR / "crop_recommendation_profiles.csv")

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["N_K_Ratio"] = enriched["N"] / (enriched["K"] + 1)
    return enriched


def get_models(feature_names: list[str], class_count: int) -> dict[str, object]:
    linear = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    solver="lbfgs",
                    C=2.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    tree = DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, random_state=RANDOM_STATE)
    forest = RandomForestClassifier(
        n_estimators=220,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    hist_gb = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=180,
        l2_regularization=0.01,
        random_state=RANDOM_STATE,
    )
    xgb = XGBClassifier(
        n_estimators=180,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    soft_voting = VotingClassifier(
        estimators=[("linear", linear), ("forest", forest), ("xgb", xgb)],
        voting="soft",
        weights=[1, 2, 2],
        n_jobs=1,
    )
    stacking = StackingClassifier(
        estimators=[("linear", linear), ("forest", forest), ("hist_gb", hist_gb), ("xgb", xgb)],
        final_estimator=LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        cv=3,
        n_jobs=1,
    )

    return {
        "Linear_LogisticRegression": linear,
        "Tree_DecisionTree": tree,
        "Tree_RandomForest": forest,
        "GradientBoosting_HistGB": hist_gb,
        "GradientBoosting_XGBoost": xgb,
        "Ensemble_SoftVoting": soft_voting,
        "Ensemble_Stacking": stacking,
    }


def evaluate_models(models: dict[str, object], X_train, X_test, y_train, y_test, classes):
    rows = []
    predictions = {}
    probabilities = {}
    fitted = {}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
    }

    for name, model in models.items():
        cv_result = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

        rows.append(
            {
                "model": name,
                "test_accuracy": accuracy_score(y_test, pred),
                "test_precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
                "test_recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
                "test_f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
                "cv_accuracy_mean": cv_result["test_accuracy"].mean(),
                "cv_accuracy_std": cv_result["test_accuracy"].std(),
                "cv_f1_weighted_mean": cv_result["test_f1_weighted"].mean(),
                "cv_f1_weighted_std": cv_result["test_f1_weighted"].std(),
            }
        )
        predictions[name] = pred
        probabilities[name] = proba
        fitted[name] = model

        report = classification_report(y_test, pred, target_names=classes, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / f"classification_report_{name}.csv")

        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(14, 12))
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(
            include_values=True, cmap="YlGnBu", xticks_rotation=60, ax=ax, colorbar=False
        )
        ax.set_title(f"Confusion Matrix: {name}")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "confusion_matrices" / f"{name}.png", dpi=220)
        plt.close(fig)

    metrics = pd.DataFrame(rows).sort_values(
        by=["test_f1_weighted", "test_accuracy", "cv_f1_weighted_mean"], ascending=False
    )
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    return metrics, predictions, probabilities, fitted


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    plot_df = metrics.melt(
        id_vars="model",
        value_vars=["test_accuracy", "test_precision_weighted", "test_recall_weighted", "test_f1_weighted"],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=plot_df, x="score", y="model", hue="metric", ax=ax)
    ax.set_xlim(0, 1.02)
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Score")
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_performance_comparison.png", dpi=220)
    plt.close(fig)


def plot_combined_confusion_matrices(predictions, y_test, classes) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()
    for ax, (name, pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, cmap="YlGnBu", cbar=False, xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.tick_params(axis="x", labelrotation=75, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes[len(predictions) :]:
        ax.axis("off")
    fig.suptitle("Confusion Matrices Across All Models", fontsize=18, y=0.995)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "all_model_confusion_matrices.png", dpi=220)
    plt.close(fig)


def plot_multiclass_curves(probabilities, y_test, classes) -> None:
    y_bin = label_binarize(y_test, classes=np.arange(len(classes)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for name, proba in probabilities.items():
        RocCurveDisplay.from_predictions(
            y_bin.ravel(),
            proba.ravel(),
            name=name.replace("_", " "),
            ax=axes[0],
            linewidth=1.7,
        )
        PrecisionRecallDisplay.from_predictions(
            y_bin.ravel(),
            proba.ravel(),
            name=name.replace("_", " "),
            ax=axes[1],
            linewidth=1.7,
        )
    axes[0].set_title("Micro-Averaged ROC Curves")
    axes[1].set_title("Micro-Averaged Precision-Recall Curves")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_precision_recall_curves.png", dpi=220)
    plt.close(fig)


def choose_best_model(metrics: pd.DataFrame) -> str:
    top_score = metrics.iloc[0]["test_f1_weighted"]
    tied = metrics[np.isclose(metrics["test_f1_weighted"], top_score)]
    shap_preference = [
        "GradientBoosting_XGBoost",
        "Tree_RandomForest",
        "GradientBoosting_HistGB",
        "Ensemble_SoftVoting",
        "Ensemble_Stacking",
        "Linear_LogisticRegression",
        "Tree_DecisionTree",
    ]
    for name in shap_preference:
        if name in tied["model"].values:
            return name
    return metrics.iloc[0]["model"]


def run_shap(best_name: str, best_model, X_train, X_test, classes, feature_names) -> pd.DataFrame:
    if best_name == "GradientBoosting_XGBoost":
        sample_test = shap.sample(X_test, min(440, len(X_test)), random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(sample_test)
    elif best_name == "Tree_RandomForest":
        sample_test = shap.sample(X_test, min(440, len(X_test)), random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(sample_test)
    else:
        sample_train = shap.sample(X_train, min(80, len(X_train)), random_state=RANDOM_STATE)
        sample_test = shap.sample(X_test, min(120, len(X_test)), random_state=RANDOM_STATE)
        explainer = shap.Explainer(best_model.predict_proba, sample_train, algorithm="permutation")
        explanation = explainer(sample_test, max_evals=2 * len(feature_names) + 1)
        shap_values = explanation.values

    values = np.asarray(shap_values)
    if values.ndim == 3 and values.shape[1] == len(feature_names):
        values_by_class = np.moveaxis(values, 2, 0)
    elif values.ndim == 3 and values.shape[2] == len(feature_names):
        values_by_class = values
    elif isinstance(shap_values, list):
        values_by_class = np.asarray(shap_values)
    else:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    records = []
    for class_index, crop in enumerate(classes):
        class_values = values_by_class[class_index]
        mean_abs = np.abs(class_values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        for rank, feature_index in enumerate(order, start=1):
            records.append(
                {
                    "crop": crop,
                    "rank": rank,
                    "feature": feature_names[feature_index],
                    "mean_abs_shap": mean_abs[feature_index],
                }
            )
    shap_table = pd.DataFrame(records)
    shap_table.to_csv(OUTPUT_DIR / "shap_feature_importance_by_crop.csv", index=False)
    shap_table[shap_table["rank"] <= 3].to_csv(OUTPUT_DIR / "top_3_shap_features_by_crop.csv", index=False)

    global_importance = (
        shap_table.groupby("feature", as_index=False)["mean_abs_shap"].mean().sort_values("mean_abs_shap")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(global_importance["feature"], global_importance["mean_abs_shap"], color="#2e7d32")
    ax.set_title(f"Global SHAP Importance: {best_name}")
    ax.set_xlabel("Mean absolute SHAP value across crops")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_global_feature_importance.png", dpi=220)
    plt.close(fig)

    top_plot = shap_table[shap_table["rank"] <= 5].copy()
    top_plot["crop"] = pd.Categorical(top_plot["crop"], categories=classes, ordered=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.scatterplot(
        data=top_plot,
        x="mean_abs_shap",
        y="crop",
        hue="feature",
        size="rank",
        sizes=(120, 35),
        ax=ax,
    )
    ax.set_title("Top SHAP Features For Each Crop")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_top_features_by_crop.png", dpi=220)
    plt.close(fig)

    return shap_table


def main() -> None:
    make_output_dir()
    raw_df = load_and_validate_dataset()
    df = add_features(raw_df)

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X = df.drop(columns=["label"])
    classes = le.classes_
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = get_models(feature_names, len(classes))
    metrics, predictions, probabilities, fitted = evaluate_models(models, X_train, X_test, y_train, y_test, classes)
    plot_model_comparison(metrics)
    plot_combined_confusion_matrices(predictions, y_test, classes)
    plot_multiclass_curves(probabilities, y_test, classes)

    best_name = choose_best_model(metrics)
    best_model = fitted[best_name]
    best_pred = predictions[best_name]
    pd.DataFrame(classification_report(y_test, best_pred, target_names=classes, output_dict=True)).transpose().to_csv(
        OUTPUT_DIR / "best_model_classification_report.csv"
    )
    joblib.dump(best_model, OUTPUT_DIR / "best_crop_model.joblib")
    joblib.dump(le, OUTPUT_DIR / "label_encoder.joblib")

    shap_table = run_shap(best_name, best_model, X_train, X_test, classes, feature_names)

    summary = {
        "dataset_rows": int(len(raw_df)),
        "dataset_columns": list(raw_df.columns),
        "classes": classes.tolist(),
        "feature_columns_used": feature_names,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_model": best_name,
        "best_model_metrics": metrics[metrics["model"] == best_name].iloc[0].to_dict(),
        "top_global_shap_features": shap_table.groupby("feature")["mean_abs_shap"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .to_dict(),
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
