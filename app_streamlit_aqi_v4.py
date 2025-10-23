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
    try:
        mr = _project.get_model_registry()
        model_meta = mr.get_model(model_name, version=version) if version else mr.get_model(model_name)
        model_dir = model_meta.download("tmp_hopsworks_model")

        for root, _, files in os.walk(model_dir):
            for f in files:
                if f.endswith((".pkl", ".joblib")):
                    return joblib.load(os.path.join(root, f))
    except Exception as e:
        pass
        # st.warning(f"Hopsworks model not available. Using local model.\n({e})")

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
    """
    Iterative 1-hour ahead forecasting using the last rows as seed.
    Returns a DataFrame with columns ['datetime','pred_aqi'] and datetime as pandas timestamps.
    """
    recent = seed_df.copy().reset_index(drop=True)
    max_lag = sum(1 for f in features if "aqi_lag_" in f)
    preds, times = [], []

    for _ in range(horizon_hours):
        row = recent.iloc[-1:].copy()
        # ensure lag features exist in the row
        for j in range(1, max_lag+1):
            # if recent has enough rows
            if len(recent) >= j:
                row[f"aqi_lag_{j}"] = recent['aqi'].iloc[-j]
            else:
                row[f"aqi_lag_{j}"] = np.nan
        # select features in the same order as features list
        X_future = row[features]
        # if model expects 2D array
        pred = model.predict(X_future)[0]
        preds.append(pred)
        next_time = row['datetime'].iloc[0] + pd.Timedelta(hours=1)
        times.append(next_time)
        new_row = {'datetime': next_time, 'aqi': pred}
        for c in ['temperature_2m', 'relative_humidity_2m', 'windspeed_10m', 'ozone']:
            if c in row.columns:
                new_row[c] = row[c].iloc[0]
        new_row.update({
            'hour': (row['hour'].iloc[0] + 1) % 24 if 'hour' in row else next_time.hour,
            'day': row['day'].iloc[0] if 'day' in row else next_time.day,
            'month': row['month'].iloc[0] if 'month' in row else next_time.month,
        })
        recent = pd.concat([recent, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame({"datetime": pd.to_datetime(times), "pred_aqi": preds})

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    return {"MAE": mae, "RMSE": rmse}

# 🆕 AQI CATEGORY INTERPRETATION (with emoji text)
def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good 😊"
    elif aqi_value <= 100:
        return "Moderate 😐"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups 😷"
    elif aqi_value <= 200:
        return "Unhealthy 😞"
    elif aqi_value <= 300:
        return "Very Unhealthy ☠️"
    else:
        return "Hazardous 💀"

# color map for status card
AQI_COLOR_MAP = {
    "Good 😊": "#2ECC71",
    "Moderate 😐": "#F1C40F",
    "Unhealthy for Sensitive Groups 😷": "#E67E22",
    "Unhealthy 😞": "#E74C3C",
    "Very Unhealthy ☠️": "#8E44AD",
    "Hazardous 💀": "#7E0023"
}

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
    # safer extraction for attributes
    if col == 'hour':
        df[col] = df['datetime'].dt.hour
    elif col == 'day':
        df[col] = df['datetime'].dt.day
    elif col == 'month':
        df[col] = df['datetime'].dt.month

model = load_model_from_registry(project, DEFAULT_MODEL_NAME, version=3) if project else None
if model is None:
    # fallback to local model if registry did not work
    if os.path.exists(LOCAL_MODEL_PATH):
        model = joblib.load(LOCAL_MODEL_PATH)
    else:
        st.error("❌ No model available (Hopsworks or local).")
        st.stop()

st.sidebar.success("✅ Data and model ready.")

# -------------------------
# 📈 Current AQI (always visible)
# -------------------------
st.markdown("## 🟢 Current AQI")

# pick latest available AQI - prefer last non-null
if 'aqi' in df.columns:
    latest_aqi_row = df[df['aqi'].notna()].tail(1)
    if not latest_aqi_row.empty:
        latest_row = latest_aqi_row.iloc[0]
        current_aqi = latest_row['aqi']
    else:
        current_aqi = float(df['aqi'].iloc[-1]) if 'aqi' in df.columns else np.nan
else:
    current_aqi = np.nan

current_status = get_aqi_category(current_aqi) if not np.isnan(current_aqi) else "No data"

card_color = AQI_COLOR_MAP.get(current_status, "#34495E")

st.markdown(
    f"""
    <div style="display:flex; gap:20px; align-items:center;">
      <div style='padding:18px;border-radius:10px;background:{card_color};min-width:220px;text-align:center;color:white;'>
        <div style='font-size:20px;font-weight:600;'>Current AQI</div>
        <div style='font-size:28px;margin-top:6px;font-weight:700;'>{current_aqi:.1f}</div>
        <div style='margin-top:6px;font-size:16px'>{current_status}</div>
      </div>
      <div style='padding:12px;'>
        <small>Last update:</small><br/>
        <strong>{pd.to_datetime(df['datetime'].iloc[-1]).strftime('%Y-%m-%d %H:%M')}</strong>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------
# 📈 Forecast Section
# -------------------------
# -------------------------
# 📈 Forecast Section
# -------------------------
st.markdown("## 🔮 Forecasting")
st.info("Generate future AQI predictions using the trained model.")

# ---- Left column for controls ----
st.subheader("Forecast Controls")

# Generate forecast button
if st.button("🚀 Generate Forecast"):
    seed_df = df.tail(lag_window).copy()
    base_features = ['temperature_2m', 'relative_humidity_2m', 'windspeed_10m', 'ozone', 'hour', 'day', 'month']
    lag_features = [f"aqi_lag_{i}" for i in range(1, lag_window+1)]
    features = [f for f in base_features + lag_features if (f in df.columns) or ("aqi_lag_" in f)]
    forecast_df = generate_iterative_forecast(model, seed_df, features, horizon_hours=horizon)
    st.session_state['forecast_df'] = forecast_df
    st.success("✅ Forecast generated!")

# ---- Main Forecast Graph ----
if 'forecast_df' in st.session_state:
    ff = st.session_state['forecast_df']

    fig = go.Figure()
    # show last 200 actuals
    if 'aqi' in df.columns:
        sample_actuals = df.tail(200)
        fig.add_trace(go.Scatter(
            x=sample_actuals['datetime'], y=sample_actuals['aqi'],
            mode='lines', name='Actual AQI'
        ))

    # forecasted AQI
    fig.add_trace(go.Scatter(
        x=ff['datetime'], y=ff['pred_aqi'],
        mode='lines+markers', name='Forecast AQI'
    ))

    fig.update_layout(
        title="📈 AQI: Future vs Forecast",
        xaxis_title="Datetime",
        yaxis_title="AQI",
        height=520
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ Click 'Generate Forecast' to view the AQI forecast graph.")

# ---- Show Next 72h AQI Details ----
if 'show_72h_details' not in st.session_state:
    st.session_state['show_72h_details'] = False

if st.button("🌤 Show Next 72h AQI Details"):
    st.session_state['show_72h_details'] = True

if st.session_state.get('show_72h_details', False):
    if 'forecast_df' not in st.session_state:
        st.warning("⚠️ No forecast found. Click 'Generate Forecast' first.")
    else:
        fdf = st.session_state['forecast_df'].copy()
        fdf['datetime'] = pd.to_datetime(fdf['datetime'])
        fdf['AQI Category'] = fdf['pred_aqi'].apply(get_aqi_category)
        fdf['date'] = fdf['datetime'].dt.date

        st.markdown("### 📅 Choose day to view")
        day_choice = st.radio("Select:", ["Today", "Tomorrow", "Day After Tomorrow"], horizontal=True)

        today = pd.Timestamp.now().date()
        if day_choice == "Today":
            filter_date = today
        elif day_choice == "Tomorrow":
            filter_date = today + timedelta(days=1)
        else:
            filter_date = today + timedelta(days=2)

        filtered = fdf[fdf['date'] == filter_date].reset_index(drop=True)
        st.markdown(f"#### 📊 Forecast for **{filter_date}**")

        if filtered.empty:
            st.warning("⚠️ No forecast data available for the selected day.")
        else:
            display_df = filtered[['datetime', 'pred_aqi', 'AQI Category']].rename(columns={'pred_aqi': 'AQI (pred)'})
            st.dataframe(display_df.style.format({'AQI (pred)': '{:.1f}'}), use_container_width=True)

            csv = display_df.to_csv(index=False)
            st.download_button("⬇️ Download Selected Day Forecast CSV", csv, file_name=f"aqi_forecast_{filter_date}.csv", mime="text/csv")

            # Daily chart
            fig_day = go.Figure()
            fig_day.add_trace(go.Scatter(x=filtered['datetime'], y=filtered['pred_aqi'], mode='lines+markers', name='Forecast AQI'))

            actuals_daily = df.set_index('datetime').resample('1H').mean().interpolate().reset_index()
            overlap = actuals_daily[(actuals_daily['datetime'].dt.date == filter_date)]
            if not overlap.empty and 'aqi' in overlap.columns:
                fig_day.add_trace(go.Scatter(x=overlap['datetime'], y=overlap['aqi'], mode='lines', name='Actual AQI'))

            fig_day.update_layout(title=f"📈 AQI Forecast vs Actual ({filter_date})", xaxis_title="Datetime", yaxis_title="AQI", height=450)
            st.plotly_chart(fig_day, use_container_width=True)

            avg_aqi = filtered['pred_aqi'].mean()
            st.info(f"🌍 Average predicted AQI for {filter_date}: **{avg_aqi:.1f}** → {get_aqi_category(avg_aqi)}")

    if st.button("Hide 72h Details"):
        st.session_state['show_72h_details'] = False

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
