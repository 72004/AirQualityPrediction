# app_streamlit_aqi.py
"""
Streamlit dashboard for AQI forecasting & EDA

Features:
- Load latest features from Hopsworks Feature Store
- Load model from Hopsworks Model Registry (or local fallback)
- Compute iterative 72-hour forecast (uses lag features)
- Interactive charts and EDA tools:
    - Time series (past + forecast)
    - Rolling means, histograms
    - Correlation heatmap
    - Feature importance (for RandomForest)
    - Simple anomaly detection (z-score)
- Sidebar controls to change lag window, forecasting horizon, and data source
"""

import os
from datetime import timedelta
import pandas as pd
import numpy as np
import joblib
import hopsworks
from dotenv import load_dotenv
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Configuration / Helpers
# -------------------------
# --- Load environment variables ---
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# --- Default constants ---
DEFAULT_FEATURE_GROUP_NAME = "weather_data_2"
DEFAULT_FG_VERSION = 1
DEFAULT_MODEL_NAME = "AQI_RF_Forecaster"
LOCAL_MODEL_PATH = "aqi_rf_model/rf_aqi_forecast.pkl"

# --- Streamlit setup ---
st.set_page_config(page_title="AQI Forecast Dashboard", layout="wide")

# --- Connect to Hopsworks ---
@st.cache_resource
def connect_hopsworks(api_key: str = None):
    """Connect to Hopsworks and return the project object."""
    try:
        if api_key:
            project = hopsworks.login(api_key_value=api_key)
        else:
            project = hopsworks.login()
        st.success("✅ Connected to Hopsworks successfully.")
        return project
    except Exception as e:
        st.error(f"❌ Could not connect to Hopsworks: {e}")
        return None

# --- Cached feature group loader ---
@st.cache_data
def load_feature_group(_project, fg_name, version=1, nrows=None):
    fg = _project.get_feature_store().get_feature_group(name=fg_name, version=version)
    df = fg.read()
    if nrows:
        df = df.tail(nrows)
    return df

# --- Initialize connection first ---
project = connect_hopsworks(HOPSWORKS_API_KEY)

if project is None:
    st.stop()  # Prevent rest of the script if connection fails

# --- Load the feature group ---
fg_name = DEFAULT_FEATURE_GROUP_NAME
fg_version = DEFAULT_FG_VERSION
nrows = 500  # Adjust as needed

try:
    df = load_feature_group(project, fg_name, version=fg_version, nrows=nrows)
    st.write("✅ Data Loaded from Feature Store:")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"❌ Failed to load feature group: {e}")
    st.stop()
    
@st.cache_resource
def load_model_from_registry(project, model_name="AQI_RF_Forecaster", version=None):
    """Load model directly from Hopsworks Model Registry."""
    try:
        mr = project.get_model_registry()

        # Get specific version if provided, else latest
        if version:
            model_meta = mr.get_model(model_name, version=version)
        else:
            model_meta = mr.get_model(model_name)

        st.info(f"📦 Found model: {model_meta.name} (version: {model_meta.version})")

        # Download model artifacts temporarily
        tmpdir = "tmp_hopsworks_model"
        model_dir = model_meta.download(tmpdir)
        st.info(f"✅ Model artifacts downloaded to: {model_dir}")

        # Look for .pkl or .joblib inside
        for root, _, files in os.walk(model_dir):
            for f in files:
                if f.endswith(".pkl") or f.endswith(".joblib"):
                    model_path = os.path.join(root, f)
                    st.success(f"🎯 Loaded model from: {model_path}")
                    return joblib.load(model_path)

        st.error("⚠️ No .pkl/.joblib file found in model artifacts.")
        return None

    except Exception as e:
        st.error(f"❌ Failed to load model from Hopsworks: {e}")
        return None
        
def calculate_aqi(pm25, pm10):
    """Simple AQI calculation used for training target."""
    return 0.5 * pm25 + 0.5 * pm10

def create_lagged_features(data, lag=24, target_col="aqi"):
    df = data.copy().sort_values("datetime").reset_index(drop=True)
    for i in range(1, lag+1):
        df[f"{target_col}_lag_{i}"] = df[target_col].shift(i)
    return df.dropna().reset_index(drop=True)

def generate_iterative_forecast(model, seed_df, features, horizon_hours=72):
    """
    seed_df must contain at least max_lag rows (most recent at the bottom).
    features should include the lag features (aqi_lag_1 ... aqi_lag_N) and static features.
    This function iteratively predicts the next horizon_hours and returns list of preds and timestamps.
    """
    recent = seed_df.copy().reset_index(drop=True)
    max_lag = sum(1 for f in features if "aqi_lag_" in f)
    preds = []
    times = []
    for i in range(horizon_hours):
        row = recent.iloc[-1:].copy()

        # ensure lag features are present (recompute from recent)
        for j in range(1, max_lag+1):
            row[f"aqi_lag_{j}"] = recent['aqi'].iloc[-j]

        X_future = row[features]
        # sklearn expects 2D array
        pred = model.predict(X_future)[0]
        preds.append(pred)

        next_time = row['datetime'].iloc[0] + pd.Timedelta(hours=1)
        times.append(next_time)

        # append new row
        new_row = {
            'datetime': next_time,
            'aqi': pred,
        }
        # copy forward static numeric features if available
        for col in ['temperature_2m', 'relative_humidity_2m', 'windspeed_10m', 'ozone']:
            if col in row.columns:
                new_row[col] = row[col].iloc[0]
        if 'hour' in row.columns:
            new_row['hour'] = (row['hour'].iloc[0] + 1) % 24
        if 'day' in row.columns:
            new_row['day'] = row['day'].iloc[0]
        if 'month' in row.columns:
            new_row['month'] = row['month'].iloc[0]

        recent = pd.concat([recent, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame({"datetime": times, "pred_aqi": preds})

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    return {"MAE": mae, "RMSE": rmse}

# -------------------------
# Streamlit UI
# -------------------------
st.title("🌤 AQI Forecasting Dashboard")
st.markdown(
    """
    This dashboard loads data and model from Hopsworks (or local fallback),
    computes iterative 72-hour forecasts, and provides EDA & analytics tools.
    """
)

# Sidebar: connection & options
st.sidebar.header("Configuration")
use_hopsworks = st.sidebar.checkbox("Use Hopsworks (requires HOPSWORKS_API_KEY)", value=True)
api_key_input = st.sidebar.text_input("Hopsworks API Key (optional override)", value="", type="password")
fg_name = st.sidebar.text_input("Feature Group Name", DEFAULT_FEATURE_GROUP_NAME)
fg_version = st.sidebar.number_input("Feature Group Version", value=DEFAULT_FG_VERSION, min_value=1, step=1)
nrows = st.sidebar.number_input("Rows to load (0 = all)", value=0, min_value=0, step=1)
lag_window = st.sidebar.selectbox("Lag window (hours) for model features", options=[12, 24, 36, 48], index=1)
horizon = st.sidebar.selectbox("Forecast horizon (hours)", options=[24, 48, 72], index=2)
load_model_btn = st.sidebar.button("(Re)load model & data")

# Connect & load data/model
project = None
if use_hopsworks:
    project = connect_hopsworks(api_key_input or HOPSWORKS_API_KEY)
    if project is None:
        st.sidebar.error("Hopsworks connection failed; switch off Hopsworks or supply key.")
else:
    st.sidebar.info("Hopsworks disabled — app will try to use local files.")

# Data source: Hopsworks or local CSV fallback
df = None
if project and use_hopsworks:
    df = load_feature_group(project, fg_name, version=fg_version, nrows=nrows if nrows>0 else None)
else:
    # local fallback: try to find a local CSV
    local_csv = "data/weather_data_2.csv"
    if os.path.exists(local_csv):
        df = pd.read_csv(local_csv)
        st.sidebar.success(f"Loaded local CSV: {local_csv}")
    else:
        st.sidebar.warning("No local CSV found. Please provide a local file at data/weather_data_2.csv or enable Hopsworks.")

if df is None or df.empty:
    st.warning("No data available to show. Connect Hopsworks or place a local CSV at data/weather_data_2.csv.")
    st.stop()

# prepare df
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# compute aqi if not present
if 'aqi' not in df.columns:
    if {'pm2_5', 'pm10'}.issubset(df.columns):
        df['aqi'] = calculate_aqi(df['pm2_5'], df['pm10'])
    else:
        st.error("Cannot compute AQI: pm2_5 and pm10 columns missing.")
        st.stop()

# Feature engineering: hour/day/month
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month

# show basic info
st.sidebar.markdown(f"**Rows:** {len(df)}")
st.sidebar.markdown(f"**Date range:** {df['datetime'].min()} — {df['datetime'].max()}")

# load model
model = None
if project and use_hopsworks:
    model = load_model_from_registry(project, DEFAULT_MODEL_NAME)
else:
    if os.path.exists(LOCAL_MODEL_PATH):
        model = joblib.load(LOCAL_MODEL_PATH)
        st.sidebar.success("Loaded model from local path.")
    else:
        st.sidebar.warning("No model found locally; please train & save model first.")

if model is None:
    st.error("Model not available. Upload model to Hopsworks or save it locally at the expected path.")
    st.stop()

# Prepare dataset with lag features for evaluation and seeding forecasts
df_lagged = create_lagged_features(df.copy(), lag=lag_window, target_col="aqi")

# allow user to select time range for EDA
st.sidebar.header("EDA Window")
eda_days = st.sidebar.slider("Lookback window (days) for EDA", min_value=1, max_value=30, value=7)
eda_df = df[df['datetime'] >= (df['datetime'].max() - pd.Timedelta(days=eda_days))].copy()

# Main layout: left = controls + metrics, right = charts
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Model & Forecast Controls")
    st.write("Latest timestamp:", df['datetime'].max())
    if st.button("Compute Forecast (now)"):
        # use last rows as seed
        seed_rows_needed = lag_window
        seed_df = df.tail(seed_rows_needed).copy()
        # ensure created lag features exist in seed_df by adding aqi column (we have it)
        # build list of features expected by the model:
        # We'll attempt to infer features from the model if possible (scikit-learn RF doesn't store feature names)
        # So we build features consistent with training script:
        base_features = ['temperature_2m', 'relative_humidity_2m', 'windspeed_10m', 'ozone', 'hour', 'day', 'month']
        lag_features = [f"aqi_lag_{i}" for i in range(1, lag_window+1)]
        features = [f for f in base_features + lag_features if f in df.columns or "aqi_lag_" in f]
        # ensure seed_df has the base cols (if missing, fill with last-known)
        for c in ['temperature_2m', 'relative_humidity_2m', 'windspeed_10m', 'ozone']:
            if c not in seed_df.columns:
                seed_df[c] = df[c].iloc[-1] if c in df.columns else 0.0
        # generate forecast
        forecast_df = generate_iterative_forecast(model, seed_df, features, horizon_hours=horizon)
        st.session_state['forecast_df'] = forecast_df
        st.success("Forecast computed and stored in session.")

    if 'forecast_df' in st.session_state:
        st.write("Last forecast computed:", st.session_state['forecast_df'].iloc[0].to_dict())
        if st.button("Clear forecast"):
            del st.session_state['forecast_df']
            st.info("Forecast cleared.")

    st.markdown("---")
    st.subheader("Quick Metrics (on last split)")
    # Attempt quick evaluation: compare model to last portion of df_lagged
    try:
        # use last 20% for quick eval
        split = int(0.8 * len(df_lagged))
        X_eval = df_lagged.iloc[split:][[c for c in df_lagged.columns if c.startswith(('temperature_2m','relative_humidity_2m','windspeed_10m','ozone','hour','day','month','aqi_lag_'))]]
        y_eval = df_lagged['aqi'].iloc[split:]
        if len(X_eval) > 0:
            y_pred_eval = model.predict(X_eval)
            metrics = compute_metrics(y_eval.values, y_pred_eval)
            st.metric("MAE", f"{metrics['MAE']:.2f}")
            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
        else:
            st.info("Not enough data for quick evaluation.")
    except Exception as e:
        st.error(f"Quick eval failed: {e}")

with col2:
    st.subheader("AQI: Past and Forecast")
    # plot past aqi and forecast if available
    past = df.tail(72*3)  # last 3 days by default (approx)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=past['datetime'], y=past['aqi'], mode='lines+markers', name='Actual AQI'))
    if 'forecast_df' in st.session_state:
        fdf = st.session_state['forecast_df']
        fig.add_trace(go.Scatter(x=fdf['datetime'], y=fdf['pred_aqi'], mode='lines+markers', name='Forecast AQI'))
    fig.update_layout(title="AQI: Recent vs Forecast", xaxis_title="Datetime", yaxis_title="AQI", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Exploratory Data Analysis (EDA)")

eda_col1, eda_col2 = st.columns([2,1])

with eda_col1:
    st.write("Time series & rolling statistics")
    window_hours = st.slider("Rolling window (hours)", min_value=3, max_value=72, value=24)
    eda_plot_df = eda_df.set_index('datetime').resample('1H').mean().interpolate()
    eda_plot_df['aqi_roll_mean'] = eda_plot_df['aqi'].rolling(window=window_hours).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=eda_plot_df.index, y=eda_plot_df['aqi'], name='AQI', mode='lines'))
    fig2.add_trace(go.Scatter(x=eda_plot_df.index, y=eda_plot_df['aqi_roll_mean'], name=f'Rolling mean ({window_hours}h)'))
    fig2.update_layout(title="AQI Time Series & Rolling Mean", xaxis_title="Datetime", yaxis_title="AQI")
    st.plotly_chart(fig2, use_container_width=True)

    st.write("Histogram of AQI (selected window)")
    fig_hist = px.histogram(eda_plot_df.reset_index(), x='aqi', nbins=40, title="AQI Distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

with eda_col2:
    st.write("Correlation heatmap")
    corr_cols = [c for c in eda_df.columns if eda_df[c].dtype in [np.float64, np.int64]]
    corr = eda_df[corr_cols].corr()
    fig_heat = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlations")
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")
st.subheader("Advanced Analytics")

adv_col1, adv_col2 = st.columns([1,1])

with adv_col1:
    st.write("Feature importance (Random Forest)")
    try:
        if hasattr(model, "feature_importances_"):
            # Build feature list similar to training expectations:
            fi_features = ['temperature_2m', 'relative_humidity_2m', 'windspeed_10m', 'ozone', 'hour', 'day', 'month'] + [f"aqi_lag_{i}" for i in range(1, lag_window+1)]
            # Clip to features present in model training (best-effort)
            fi_features = [f for f in fi_features if True]  # we display top n
            importances = model.feature_importances_
            # If lengths mismatch, create generic names
            if len(importances) == len(fi_features):
                fi_df = pd.DataFrame({"feature": fi_features, "importance": importances}).sort_values("importance", ascending=False).head(30)
            else:
                # fallback generic names
                fi_df = pd.DataFrame({"feature": [f"f{i}" for i in range(len(importances))], "importance": importances}).sort_values("importance", ascending=False).head(30)
            fig_bar = px.bar(fi_df, x='importance', y='feature', orientation='h', title='Feature importance (top features)')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Model doesn't expose feature_importances_.")
    except Exception as e:
        st.error(f"Feature importance failed: {e}")

with adv_col2:
    st.write("Anomaly detection (simple z-score on AQI)")
    z_thresh = st.slider("Z-score threshold", min_value=1.0, max_value=5.0, value=3.0)
    eda_df['aqi_z'] = (eda_df['aqi'] - eda_df['aqi'].mean()) / eda_df['aqi'].std()
    anomalies = eda_df[np.abs(eda_df['aqi_z']) > z_thresh]
    st.write(f"Found {len(anomalies)} anomalies in selected window (|z| > {z_thresh})")
    if len(anomalies) > 0:
        st.dataframe(anomalies[['datetime','aqi','aqi_z']].sort_values('datetime'), height=200)

st.markdown("---")
st.subheader("Data & Download")

st.write("Preview of most recent data")
st.dataframe(df.tail(200), height=300)

# allow CSV download of forecast if available
if 'forecast_df' in st.session_state:
    csv = st.session_state['forecast_df'].to_csv(index=False)
    st.download_button("Download forecast CSV", csv, file_name="aqi_forecast.csv", mime="text/csv")

st.markdown("---")
st.write("Tips & next steps:")
st.write("""
- If you want more advanced explanations use SHAP (install shap) to show per-prediction explanations.
- For production real-time prediction, consider exposing a FastAPI endpoint that the UI (or other services) can call.
- Add authentication if deploying publicly.
""")
