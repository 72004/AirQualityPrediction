# AQI Forecasting System (Serverless End-to-End Pipeline)

## Objective
Predict the Air Quality Index (AQI) in Karachi for the next 3 days (72 hours) using a 100% serverless stack.  
This project demonstrates an end-to-end machine learning pipeline for AQI forecasting with:

- Automated data collection  
- Feature engineering  
- Model training and registry management  
- Real-time AQI prediction via Streamlit dashboard  

---

## Project Architecture
The project runs entirely serverless using:

- Open-Meteo API – Weather and pollutant data  
- Hopsworks Feature Store – Data storage & model registry  
- GitHub Actions – Pipeline automation  
- Streamlit – Interactive web dashboard  

---

## Repository Structure

| File/Folder | Description |
|--------------|-------------|
| `.github/workflows/` | Contains GitHub Actions workflows for automating ingestion and training. |
| `open_metro_1_year.py` | Fetches one year of historical weather & pollutant data (Oct 2024–Oct 2025) from Open-Meteo API and saves it as a CSV file. |
| `open_metro_hopswork_pushed_v3.py` | Pushes the generated CSV to Hopsworks Feature Store and periodically fetches new hourly weather and pollutant records. Automated via GitHub Actions (runs hourly). |
| `AQI_Model_training_v2.py` | Pulls data from Hopsworks, trains a Random Forest model for 72-hour AQI forecasting, and stores the model in both Hopsworks Model Registry and locally in `aqi_model_push/`. Automated via GitHub Actions (runs every 12 hours). |
| `app_streamlit_aqi_v6.py` | Streamlit web app that loads the trained model and displays the current AQI and predicted AQI for the next 72 hours. |
| `karachi_weather_pollutants_2024_2025.csv` | Historical dataset generated from the Open-Meteo API. |
| `aqi_model_push/` | Contains locally saved trained models. |
| `requirements.txt` | List of dependencies required for running the project. |
| `.gitignore` | Specifies files and folders ignored by Git. |
| `.gitattributes` | Configures Git LFS (for tracking large files). |

---

## Automation Overview

| Process | Script | Frequency | Trigger |
|----------|---------|------------|----------|
| Data Ingestion | `open_metro_hopswork_pushed_v3.py` | Every 1 hour | GitHub Action |
| Model Training | `AQI_Model_training_v2.py` | Every 12 hours | GitHub Action |

---

## Technologies Used
- Python  
- Hopsworks (Feature Store + Model Registry)  
- Open-Meteo API  
- Streamlit  
- scikit-learn (Random Forest)  
- GitHub Actions (CI/CD)  

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/72004/AirQualityPrediction.git
cd AirQualityPrediction

# 2. Install dependencies and run the Streamlit app
pip install -r requirements.txt
streamlit run app_streamlit_aqi_v6.py

```
## 🌍 Output

The Streamlit dashboard displays:

- 📈 **Current AQI value**  
- 🌤️ **72-hour AQI forecast (next 3 days)**  
- 📊 **Visualized AQI trends and insights**

