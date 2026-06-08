"""
app.py  --  interactive demo for the Forest Cover Type classifier.

Run:  streamlit run app.py
(You must run `python run_all.py` first so that model.joblib exists.)

A user sets the cartographic variables on the left, and the model predicts
which of the 7 forest cover types is most likely, with class probabilities.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

import features as F

st.set_page_config(page_title="Forest Cover Type Predictor", page_icon="🌲", layout="wide")
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


st.title("🌲 Forest Cover Type Predictor")
st.caption(
    "Predicts the dominant tree species in a 30×30 m patch of the Roosevelt "
    "National Forest from cartographic variables — no remote sensing required."
)

if not MODEL_PATH.exists():
    st.error("`model.joblib` not found. Run `python run_all.py` first, then reload.")
    st.stop()

bundle = load_bundle()
model = bundle["model"]
feat_cols = bundle["feature_columns"]
cover_types = bundle["cover_types"]

# --- Inputs ----------------------------------------------------------------- #
with st.sidebar:
    st.header("Cartographic inputs")
    vals = {}
    for name, (lo, hi, default) in F.CONTINUOUS_RANGES.items():
        vals[name] = st.slider(name.replace("_", " "), float(lo), float(hi),
                               float(default))
    wilderness = st.selectbox(
        "Wilderness area",
        options=[1, 2, 3, 4],
        format_func=lambda i: {1: "Rawah", 2: "Neota", 3: "Comanche Peak",
                               4: "Cache la Poudre"}[i])
    soil = st.selectbox("Soil type (1–40)", options=list(range(1, 41)), index=28)

# --- Build the raw 54-column row, then engineer features -------------------- #
row = {c: 0 for c in F.RAW_COLUMNS}
row.update(vals)
row[f"Wilderness_Area{wilderness}"] = 1
row[f"Soil_Type{soil}"] = 1
raw = pd.DataFrame([row])[F.RAW_COLUMNS]
X = F.add_engineered_features(raw).reindex(columns=feat_cols, fill_value=0)

pred = int(model.predict(X)[0])
proba = model.predict_proba(X)[0]
classes = list(model.classes_)

# --- Output ----------------------------------------------------------------- #
left, right = st.columns([1, 1.2])
with left:
    st.subheader("Prediction")
    st.metric("Most likely cover type", cover_types[pred])
    st.write(f"Confidence: **{proba[classes.index(pred)]:.1%}**")
    st.caption("Tip: raise Elevation past ~3,200 m and the prediction shifts "
               "toward Krummholz / Spruce-Fir — the single strongest signal in the data.")

with right:
    st.subheader("Class probabilities")
    prob_df = (pd.DataFrame({"cover_type": [cover_types[c] for c in classes],
                             "probability": proba})
               .sort_values("probability", ascending=False)
               .set_index("cover_type"))
    st.bar_chart(prob_df)

with st.expander("What is the model?"):
    st.markdown(
        "- **Model:** histogram-based gradient-boosted decision trees "
        "(`HistGradientBoostingClassifier`), trained with balanced sample weights.\n"
        "- **Features:** the 54 cartographic variables plus 6 engineered features "
        "(straight-line distance to water, illumination summaries, elevation×slope, …).\n"
        "- **Data:** UCI / scikit-learn Covertype, 581,012 patches, 7 classes.")
