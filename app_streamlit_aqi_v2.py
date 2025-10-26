# app_streamlit_aqi_v2.py
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------
# 🌍 Environment & Config
# -------------------------
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

DEFAULT_FEATURE_GROUP_NAME = "weather_data_2"
DEFAULT_FG_VERSION = 1
DEFAULT_MODEL_NAME = "AQI_RF_Forecaster"
LOCAL_MODEL_PATH = r"C:\Weather_project\models\AQI_RF_Forecaster_latest\rf_aqi_forecast.pkl"

# -------------------------
# 🎨 Streamlit Page Setup
# -------------------------
st.set_page_config(page_title="🌤 AQI Forecast Dashboard", layout="wide", page_icon="🌤")
st.title("🌤 Air Quality Index (AQI) Forecasting Dashboard")
st.caption("📊 Powered by Random Forest | Hopsworks Feature Store | Streamlit UI")

st.markdown("---")

# -------------------------
# 🧩 Utility Functions
# -------------------------
def connect_hopsworks(api_key: str = None):
    """Connect to Hopsworks and return the project object."""
    try:
        project = hopsworks.login(api_key_value=api_key) if api_key else hopsworks.login()
        return project
    except Exception as e:
        raise RuntimeError(f"Could not connect to Hopsworks ❌\n{e}")

@st.cache_data
def load_feature_group(_project, fg_name, version=1, nrows=None):
    fg = _project.get_feature_store().get_feature_group(name=fg_name, version=version)
    df = fg.read()
    return df.tail(nrows) if nrows else df

@st.cache_resource
def load_model_from_registry(_project, model_name="AQI_RF_Forecaster", version=None):
    """Load model from Hopsworks Model Registry or fallback locally."""
    try:
        mr = _project.get_model_registry()
        model_meta = mr.get_model(model_name, version=version) if version else mr.get_model(model_name)
        model_dir = model_meta.download("tmp_hopsworks_model")

        for root, _, files in os.walk(model_dir):
            for f in files:
                if f.endswith((".pkl", ".joblib")):
                    return joblib.load(os.path.join(root, f))
    except Exception as e:
        st.warning(f"Hopsworks model not available. Using local model.\n({e})")

    if os.path.exists(LOCAL_MODEL_PATH):
        return joblib.load(LOCAL_MODEL_PATH)
    else:
        st.error("❌ No model found locally or on Hopsworks.")
        return None

def calculate_aqi(pm25, pm10):
    return 0.5 * pm25 + 0.5 * pm10

def create_lagged_features(data, lag=24, target_col="aqi"):
    df = data.copy().sort_values("datetime").reset_index(drop=True)
    for i in range(1, lag+1):
        df[f"{target_col}_lag_{i}"] = df[target_col].shift(i)
    return df.dropna().reset_index(drop=True)

def generate_iterative_forecast(model, seed_df, features, horizon_hours=72):
    recent = seed_df.copy().reset_index(drop=True)
    max_lag = sum(1 for f in features if "aqi_lag_" in f)
    preds, times = [], []

    for _ in range(horizon_hours):
        row = recent.iloc[-1:].copy()
        for j in range(1, max_lag+1):
            row[f"aqi_lag_{j}"] = recent['aqi'].iloc[-j]
        X_future = row[features]
        pred = model.predict(X_future)[0]
        preds.append(pred)
        next_time = row['datetime'].iloc[0] + pd.Timedelta(hours=1)
        times.append(next_time)
        new_row = {'datetime': next_time, 'aqi': pred}
        for c in ['temperature_2m', 'relative_humidity_2m', 'windspeed_10m', 'ozone']:
            if c in row.columns:
                new_row[c] = row[c].iloc[0]
        new_row.update({
            'hour': (row['hour'].iloc[0] + 1) % 24,
            'day': row['day'].iloc[0],
            'month': row['month'].iloc[0],
        })
        recent = pd.concat([recent, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame({"datetime": times, "pred_aqi": preds})

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    return {"MAE": mae, "RMSE": rmse}


# -------------------------
# ⚙️ Sidebar Controls
# -------------------------
st.sidebar.header("⚙️ Configuration")
use_hopsworks = st.sidebar.checkbox("Use Hopsworks", value=True)
api_key_input = st.sidebar.text_input("🔑 API Key (optional override)", value="", type="password")
fg_name = st.sidebar.text_input("📦 Feature Group", DEFAULT_FEATURE_GROUP_NAME)
fg_version = st.sidebar.number_input("Version", value=DEFAULT_FG_VERSION, min_value=1)
nrows = st.sidebar.number_input("Rows to Load (0 = all)", value=0, min_value=0)
lag_window = st.sidebar.selectbox("Lag Window (hrs)", [12, 24, 36, 48], index=1)
horizon = st.sidebar.selectbox("Forecast Horizon (hrs)", [24, 48, 72], index=2)


# -------------------------
# 🧠 Data + Model Loading
# -------------------------
project = None
if use_hopsworks:
    try:
        project = connect_hopsworks(api_key_input or HOPSWORKS_API_KEY)
        st.toast("✅ Connected to Hopsworks successfully", icon="✅")
    except Exception as e:
        st.error(str(e))

df = load_feature_group(project, fg_name, version=fg_version, nrows=nrows or None) if project else None

if df is None:
    st.warning("⚠️ Falling back to local CSV...")
    if os.path.exists("data/weather_data_2.csv"):
        df = pd.read_csv("data/weather_data_2.csv")
    else:
        st.error("❌ No data available. Provide local CSV or connect to Hopsworks.")
        st.stop()

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
if 'aqi' not in df and {'pm2_5', 'pm10'}.issubset(df.columns):
    df['aqi'] = calculate_aqi(df['pm2_5'], df['pm10'])

for col in ['hour', 'day', 'month']:
    df[col] = getattr(df['datetime'].dt, col)

model = load_model_from_registry(project, DEFAULT_MODEL_NAME, version=3) if project else joblib.load(LOCAL_MODEL_PATH)
if model is None:
    st.stop()

st.sidebar.success("✅ Data and model ready.")

# -------------------------
# 📈 Forecast Section
# -------------------------
st.markdown("## 🔮 Forecasting")
st.info("Generate future AQI predictions using the trained model.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Forecast Controls")
    if st.button("🚀 Generate Forecast"):
        seed_df = df.tail(lag_window).copy()
        base_features = ['temperature_2m', 'relative_humidity_2m', 'windspeed_10m', 'ozone', 'hour', 'day', 'month']
        lag_features = [f"aqi_lag_{i}" for i in range(1, lag_window+1)]
        features = [f for f in base_features + lag_features if f in df.columns or "aqi_lag_" in f]
        forecast_df = generate_iterative_forecast(model, seed_df, features, horizon_hours=horizon)
        st.session_state['forecast_df'] = forecast_df
        st.success("✅ Forecast generated!")

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.tail(200)['datetime'], y=df.tail(200)['aqi'],
                             mode='lines', name='Actual AQI'))
    if 'forecast_df' in st.session_state:
        fdf = st.session_state['forecast_df']
        fig.add_trace(go.Scatter(x=fdf['datetime'], y=fdf['pred_aqi'],
                                 mode='lines+markers', name='Forecast AQI'))
    fig.update_layout(title="📈 AQI: Actual vs Forecast", xaxis_title="Datetime", yaxis_title="AQI")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 📊 EDA (Collapsible)
# -------------------------
with st.expander("📊 Exploratory Data Analysis (EDA)", expanded=False):
    st.markdown("#### AQI Time Series & Rolling Mean")
    window = st.slider("Rolling Window (hrs)", 3, 72, 24)
    df_eda = df.set_index('datetime').resample('1H').mean().interpolate()
    df_eda['aqi_roll'] = df_eda['aqi'].rolling(window).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_eda.index, y=df_eda['aqi'], name='AQI'))
    fig.add_trace(go.Scatter(x=df_eda.index, y=df_eda['aqi_roll'], name='Rolling Mean'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Distribution & Correlation")
    st.plotly_chart(px.histogram(df_eda.reset_index(), x='aqi', nbins=40, title="AQI Distribution"), use_container_width=True)
    numeric_cols = df_eda.select_dtypes(include=[np.number])
    st.plotly_chart(px.imshow(numeric_cols.corr(), text_auto=True, aspect="auto", title="Feature Correlation Heatmap"), use_container_width=True)

# -------------------------
# 🧠 Advanced Analytics
# -------------------------
with st.expander("🧠 Advanced Analytics"):
    st.markdown("#### Feature Importance (Random Forest)")
    if hasattr(model, "feature_importances_"):
        feats = [f"aqi_lag_{i}" for i in range(1, lag_window+1)] + ['temperature_2m','relative_humidity_2m','windspeed_10m','ozone','hour','day','month']
        fi_df = pd.DataFrame({
            "Feature": feats[:len(model.feature_importances_)],
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)
        st.plotly_chart(px.bar(fi_df, x='Importance', y='Feature', orientation='h'), use_container_width=True)
    else:
        st.info("Model doesn’t expose feature importances.")

# -------------------------
# 💾 Data & Downloads
# -------------------------
with st.expander("💾 Data & Downloads"):
    st.dataframe(df.tail(200), height=300)
    if 'forecast_df' in st.session_state:
        csv = st.session_state['forecast_df'].to_csv(index=False)
        st.download_button("⬇️ Download Forecast CSV", csv, file_name="aqi_forecast.csv", mime="text/csv")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Hopsworks + Scikit-learn")

