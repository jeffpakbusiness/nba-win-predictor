# app/app.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------- paths ----------
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
MODEL = BASE / "model"
IMAGES = BASE / "images"

st.set_page_config(page_title="NBA Win% Predictor", page_icon="🏀", layout="centered")
st.title("🏀 NBA Win% Predictor")

# ---------- diagnostics ----------
required = {
    "model":   MODEL / "win_predictor.pkl",
    "scaler":  MODEL / "scaler.pkl",
    "features":MODEL / "features.csv",
    "metrics": MODEL / "metrics.json",
    "clean":   DATA  / "clean_team_stats.csv",
}

missing = [f"- {k}: {p}" for k, p in required.items() if not p.exists()]
st.caption(f"Repo root: `{BASE}`")
if missing:
    st.error("Missing required files:\n" + "\n".join(missing))
    st.info("Generate them by running notebooks 02 & 03, then restart the app.")
    st.stop()

# ---------- safe loads (with error display) ----------
try:
    model = joblib.load(required["model"])
    scaler = joblib.load(required["scaler"])
    features = pd.read_csv(required["features"], header=None)[0].tolist()
    metrics = json.loads(required["metrics"].read_text())

    df = pd.read_csv(required["clean"])
    # If a feature is not in the CSV (rare), drop it from the list to keep app running
    features = [f for f in features if f in df.columns]
    stats = df[features].describe().T
except Exception as e:
    st.exception(e)
    st.stop()

# --- UI ---
st.set_page_config(page_title="NBA Win% Predictor", page_icon="🏀", layout="centered")
st.title("🏀 NBA Win% Predictor")

st.caption(
    "Model: Linear Regression on team advanced stats. "
    f"Holdout metrics → **R² {metrics['r2']:.3f}**, **RMSE {metrics['rmse']:.3f}**, "
    f"baseline RMSE {metrics['baseline_rmse']:.3f}."
)

# Normalize team names so 'Atlanta Hawks*' matches 'Atlanta Hawks'
df["team_norm"] = df["team"].astype(str).str.replace("*", "", regex=False).str.strip()
team_options   = sorted(df["team_norm"].unique().tolist())
season_options = sorted(df["season"].unique().tolist())
default_season = max(season_options)

mode = st.radio("How do you want to provide inputs?", ["Pick team & season", "Enter manually"])

if mode == "Pick team & season":
    c1, c2 = st.columns(2)
    team = c1.selectbox("Team", team_options, index=team_options.index(team_options[0]))
    season = c2.selectbox("Season", season_options, index=season_options.index(default_season))

    row = df[(df["team_norm"] == team) & (df["season"] == season)]

    if row.empty:
        st.warning("No row found for that team/season. Try a different season or use 'Enter manually'.")
        st.stop()

    X = row[features].iloc[[0]].copy()
else:
    st.subheader("Manual feature inputs")
    values = {}
    for f in features:
        s = stats.loc[f]
        default = float(s["mean"])
        minv = float(s["min"]); maxv = float(s["max"])
        rng = maxv - minv
        minv = minv - 0.1 * rng; maxv = maxv + 0.1 * rng
        values[f] = st.number_input(f, value=round(default, 4),
                                    min_value=float(minv), max_value=float(maxv),
                                    step=0.001, format="%.4f")
    X = pd.DataFrame([values], columns=features)

if st.button("Predict"):
    Xs = scaler.transform(X[features])
    pred = float(model.predict(Xs)[0])    # Win %
    wins = pred * 82
    c1, c2 = st.columns(2)
    c1.metric("Predicted Win%", f"{pred:.3f}")
    c2.metric("Predicted Wins (82 gms)", f"{wins:.1f}")
    with st.expander("Show inputs"):
        st.dataframe(X.reset_index(drop=True))
