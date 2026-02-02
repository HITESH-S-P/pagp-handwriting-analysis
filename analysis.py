import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
import tempfile

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import extract_v3
import categorize

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TRAITS = [
    "Emotional Stability",
    "Mental Energy / Will Power",
    "Modesty",
    "Personal Harmony",
    "Poor Concentration",
    "Social Isolation"
]

FEATURE_GROUPS = {
    "t1": ["baseline_angle", "slant_angle"],
    "t2": ["letter_size", "pen_pressure"],
    "t3": ["letter_size", "top_margin"],
    "t4": ["line_spacing", "word_spacing"],
    "t6": ["letter_size", "line_spacing"],
    "t8": ["line_spacing", "word_spacing"]
}

# ────────────────────────────────────────────────
# Data Loading
# ────────────────────────────────────────────────
def load_data(label_file="label_list"):
    if not os.path.isfile(label_file):
        st.error(f"Dataset '{label_file}' not found.")
        return None, None

    data = pd.read_csv(label_file, sep=r"\s+", header=None)

    cols = [
        "baseline_angle",
        "top_margin",
        "letter_size",
        "line_spacing",
        "word_spacing",
        "pen_pressure",
        "slant_angle"
    ]

    features_df = data.iloc[:, :7]
    features_df.columns = cols

    labels_df = data.iloc[:, 7:15]
    labels_df.columns = [f"t{i}" for i in range(1, 9)]

    X_map = {k: features_df[FEATURE_GROUPS[k]].values for k in FEATURE_GROUPS}
    y_map = {k: labels_df[k].values for k in FEATURE_GROUPS}

    return X_map, y_map

# ────────────────────────────────────────────────
# Model Training
# ────────────────────────────────────────────────
@st.cache_resource
def train_and_load_models(force=False):
    X_data, y_data = load_data()
    if X_data is None:
        return {}, {}

    models = {}
    stats = {}

    for key in FEATURE_GROUPS:
        path = MODELS_DIR / f"lr_final_{key}.joblib"

        if path.exists() and not force:
            model = joblib.load(path)
        else:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                ))
            ])
            model.fit(X_data[key], y_data[key])
            joblib.dump(model, path)

        models[key] = model

        scores = cross_val_score(
            model,
            X_data[key],
            y_data[key],
            cv=5,
            scoring="roc_auc"
        )
        stats[key] = np.mean(scores)

    return models, stats

# ────────────────────────────────────────────────
# Feature Extraction
# ────────────────────────────────────────────────
def get_live_features(image_path):
    raw = extract_v3.start(image_path)

    mapping = [
        ("baseline_angle", categorize.determine_baseline_angle),
        ("top_margin", categorize.determine_top_margin),
        ("letter_size", categorize.determine_letter_size),
        ("line_spacing", categorize.determine_line_spacing),
        ("word_spacing", categorize.determine_word_spacing),
        ("pen_pressure", categorize.determine_pen_pressure),
        ("slant_angle", categorize.determine_slant_angle)
    ]

    feats = {}
    for i, (name, func) in enumerate(mapping):
        val, _ = func(raw[i])
        feats[name] = float(val)

    return feats

# ────────────────────────────────────────────────
# Prediction Engine
# ────────────────────────────────────────────────
def generate_predictions(models, feats):
    rows = []

    for key, trait in zip(FEATURE_GROUPS, TRAITS):
        x = np.array([[feats[f] for f in FEATURE_GROUPS[key]]])
        prob = models[key].predict_proba(x)[0][1]

        rows.append({
            "Trait": trait,
            "Score": prob,
            "Interpretation": (
                "High" if prob > 0.7 else
                "Average" if prob > 0.4 else
                "Low"
            )
        })

    return pd.DataFrame(rows)

# ────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────
def main():
    st.set_page_config("Graphology AI Statistics", layout="wide")
    st.title("Handwriting Analysis Dashboard")

    with st.sidebar:
        st.header("Controls")
        retrain = st.button("Force Retrain Models")
        if st.button("Reset Session"):
            st.cache_resource.clear()
            st.rerun()

    models, stats = train_and_load_models(force=retrain)

    st.sidebar.subheader("Model ROC-AUC")
    for i, key in enumerate(FEATURE_GROUPS):
        st.sidebar.write(f"{TRAITS[i]}: {stats[key]:.2f}")

    uploaded = st.file_uploader(
        "Upload handwriting image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            with st.status("Analyzing handwriting..."):
                feats = get_live_features(tmp_path)
                st.write("Extracted Features", feats)

                df = generate_predictions(models, feats)

            col1, col2 = st.columns(2)

            with col1:
                st.image(uploaded, caption="Input Sample", use_container_width=True)

            with col2:
                fig = px.line_polar(
                    df,
                    r="Score",
                    theta="Trait",
                    line_close=True,
                    range_r=[0, 1],
                    title="Trait Intensity Map"
                )
                fig.update_traces(fill="toself")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Detailed Metrics")
            cols = st.columns(3)
            for i, row in df.iterrows():
                with cols[i % 3]:
                    st.metric(
                        row["Trait"],
                        f"{row['Score']:.1%}",
                        row["Interpretation"]
                    )
                    st.progress(row["Score"])

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            os.unlink(tmp_path)

if __name__ == "__main__":
    main()
