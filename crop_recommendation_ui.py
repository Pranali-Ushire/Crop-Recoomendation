#.\.venv\Scripts\python.exe -m streamlit run crop_recommendation_ui.py
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "crop_analysis_outputs"
MODEL_DIR = OUTPUT_DIR / "top3_models"
PROFILE_PATH = OUTPUT_DIR / "crop_recommendation_profiles.csv"
SHAP_PATH = OUTPUT_DIR / "top_3_shap_features_by_crop.csv"
METADATA_PATH = MODEL_DIR / "metadata.json"

FEATURE_LABELS = {
    "N": "Nitrogen",
    "P": "Phosphorus",
    "K": "Potassium",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "Soil pH",
    "rainfall": "Rainfall",
    "N_K_Ratio": "Nitrogen to Potassium balance",
}


@st.cache_data
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_profiles() -> pd.DataFrame:
    return pd.read_csv(PROFILE_PATH)


@st.cache_data
def load_shap_table() -> pd.DataFrame:
    return pd.read_csv(SHAP_PATH)


@st.cache_resource
def load_model(model_name: str):
    metadata = load_metadata()
    artifact = metadata["models"][model_name]["artifact"]
    return joblib.load(MODEL_DIR / artifact)


@st.cache_resource
def load_encoder():
    return joblib.load(MODEL_DIR / "label_encoder.joblib")


def build_input_frame(values: dict[str, float], feature_columns: list[str]) -> pd.DataFrame:
    row = {
        "N": values["N"],
        "P": values["P"],
        "K": values["K"],
        "temperature": values["temperature"],
        "humidity": values["humidity"],
        "ph": values["ph"],
        "rainfall": values["rainfall"],
    }
    row["N_K_Ratio"] = row["N"] / (row["K"] + 1)
    return pd.DataFrame([[row[column] for column in feature_columns]], columns=feature_columns)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def prediction_table(model, input_df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    probabilities = model.predict_proba(input_df)[0]
    result = pd.DataFrame({"crop": classes, "confidence": probabilities})
    return result.sort_values("confidence", ascending=False).reset_index(drop=True)


def crop_profile_card(profiles: pd.DataFrame, crop: str) -> pd.DataFrame:
    profile = profiles[profiles["label"] == crop].iloc[0]
    rows = [
        ("Nitrogen", profile["N_median"]),
        ("Phosphorus", profile["P_median"]),
        ("Potassium", profile["K_median"]),
        ("Temperature", profile["temperature_median"]),
        ("Humidity", profile["humidity_median"]),
        ("Soil pH", profile["ph_median"]),
        ("Rainfall", profile["rainfall_median"]),
    ]
    return pd.DataFrame(rows, columns=["Field factor", "Typical value"])


def show_metric_row(model_info: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Accuracy", format_percent(model_info["accuracy"]))
    cols[1].metric("Precision", format_percent(model_info["precision_weighted"]))
    cols[2].metric("Recall", format_percent(model_info["recall_weighted"]))
    cols[3].metric("F1 score", format_percent(model_info["f1_weighted"]))


def main() -> None:
    st.set_page_config(page_title="Crop Recommendation Advisor", page_icon="CA", layout="wide")

    metadata = load_metadata()
    profiles = load_profiles()
    shap_table = load_shap_table()
    encoder = load_encoder()

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e6e8eb;
            padding: 14px 16px;
            border-radius: 8px;
        }
        .result-title {
            font-size: 48px;
            font-weight: 800;
            line-height: 1.05;
            color: #246b45;
            margin: 0 0 4px 0;
        }
        .muted { color: #5f6b63; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.9, 1.35], gap="large")

    with left:
        st.title("Crop Recommendation Advisor")
        st.caption("Compare the top three trained models and run a fresh crop recommendation.")

        model_options = list(metadata["models"].keys())
        model_name = st.selectbox(
            "Choose model",
            model_options,
            format_func=lambda key: metadata["models"][key]["display_name"],
        )
        show_metric_row(metadata["models"][model_name])

        st.subheader("Soil nutrients")
        c1, c2, c3 = st.columns(3)
        n = c1.number_input("Nitrogen", min_value=0.0, max_value=150.0, value=80.0, step=1.0)
        p = c2.number_input("Phosphorus", min_value=0.0, max_value=150.0, value=45.0, step=1.0)
        k = c3.number_input("Potassium", min_value=0.0, max_value=220.0, value=40.0, step=1.0)

        st.subheader("Field conditions")
        temperature = st.slider("Temperature", 0.0, 50.0, 25.0, 0.1)
        humidity = st.slider("Humidity", 0.0, 100.0, 80.0, 0.1)
        ph = st.slider("Soil pH", 3.0, 10.0, 6.5, 0.1)
        rainfall = st.number_input("Rainfall", min_value=0.0, max_value=350.0, value=150.0, step=1.0)

    values = {
        "N": n,
        "P": p,
        "K": k,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }
    model = load_model(model_name)
    input_df = build_input_frame(values, metadata["feature_columns"])
    crop_scores = prediction_table(model, input_df, encoder.classes_.tolist())
    predicted_crop = crop_scores.iloc[0]["crop"]
    confidence = crop_scores.iloc[0]["confidence"]
    top_three = crop_scores.head(3).copy()

    with right:
        st.markdown("### Recommendation")
        st.markdown(f"<p class='result-title'>{predicted_crop.title()}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p class='muted'>Selected model: <b>{metadata['models'][model_name]['display_name']}</b> | Confidence: <b>{format_percent(confidence)}</b></p>",
            unsafe_allow_html=True,
        )

        probability_fig = px.bar(
            top_three.sort_values("confidence"),
            x="confidence",
            y="crop",
            orientation="h",
            text=top_three.sort_values("confidence")["confidence"].map(lambda x: f"{x * 100:.1f}%"),
            labels={"confidence": "Confidence", "crop": "Crop"},
            color="crop",
            color_discrete_sequence=["#2e7d32", "#4f8f6b", "#9bb36d"],
        )
        probability_fig.update_layout(
            showlegend=False,
            height=260,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_tickformat=".0%",
            yaxis_title="",
        )
        st.plotly_chart(probability_fig, use_container_width=True)

        tab_result, tab_explain, tab_models = st.tabs(["Crop profile", "Why this crop", "Model comparison"])

        with tab_result:
            st.dataframe(crop_profile_card(profiles, predicted_crop), hide_index=True, use_container_width=True)

            profile = profiles[profiles["label"] == predicted_crop].iloc[0]
            radar = go.Figure()
            radar.add_trace(
                go.Scatterpolar(
                    r=[
                        values["N"] / max(profile["N_median"], 1),
                        values["P"] / max(profile["P_median"], 1),
                        values["K"] / max(profile["K_median"], 1),
                        values["temperature"] / max(profile["temperature_median"], 1),
                        values["humidity"] / max(profile["humidity_median"], 1),
                        values["rainfall"] / max(profile["rainfall_median"], 1),
                    ],
                    theta=["N", "P", "K", "Temp", "Humidity", "Rainfall"],
                    fill="toself",
                    name="Your field vs typical crop profile",
                    line_color="#246b45",
                )
            )
            radar.update_layout(
                height=360,
                margin=dict(l=20, r=20, t=30, b=20),
                polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
                showlegend=False,
            )
            st.plotly_chart(radar, use_container_width=True)

        with tab_explain:
            crop_shap = shap_table[shap_table["crop"] == predicted_crop].copy()
            crop_shap["Feature"] = crop_shap["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
            crop_shap["Influence"] = crop_shap["mean_abs_shap"]
            explain_fig = px.bar(
                crop_shap.sort_values("Influence"),
                x="Influence",
                y="Feature",
                orientation="h",
                color="Feature",
                color_discrete_sequence=["#246b45", "#7f9f51", "#c58f2a"],
            )
            explain_fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(explain_fig, use_container_width=True)

            strongest = crop_shap.sort_values("Influence", ascending=False).iloc[0]["Feature"]
            st.info(f"For {predicted_crop.title()}, the strongest driver in the saved SHAP analysis is {strongest}.")

        with tab_models:
            comparison = pd.DataFrame(metadata["models"]).transpose().reset_index(names="model")
            comparison["model"] = comparison["model"].str.replace("_", " ")
            for column in ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]:
                comparison[column] = comparison[column].astype(float)

            comparison_fig = px.bar(
                comparison.sort_values("f1_weighted"),
                x="f1_weighted",
                y="display_name",
                orientation="h",
                text=comparison.sort_values("f1_weighted")["f1_weighted"].map(lambda x: f"{x * 100:.2f}%"),
                labels={"f1_weighted": "Weighted F1 score", "display_name": "Model"},
                color="display_name",
                color_discrete_sequence=["#2e7d32", "#6b8f71", "#b8a44c"],
            )
            comparison_fig.update_layout(
                showlegend=False,
                height=280,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_tickformat=".0%",
                yaxis_title="",
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
            st.dataframe(
                comparison[["display_name", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]],
                hide_index=True,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
