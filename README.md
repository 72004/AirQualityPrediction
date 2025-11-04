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
## Implementation

The system is implemented as a fully automated, serverless pipeline designed for continuous AQI prediction and model improvement.  
The core components include:

### 1. Data Collection

- Weather and pollutant data (such as temperature, humidity, PM2.5, PM10, NO₂, and CO levels) are collected using the Open-Meteo API.

- The script open_metro_1_year.py is used to fetch historical data for one full year (Oct 2024 – Oct 2025).

- The resulting dataset is stored locally as karachi_weather_pollutants_2024_2025.csv.

### 2. Data Ingestion and Feature Store Management

- The script open_metro_hopswork_pushed_v3.py pushes the dataset to the Hopsworks Feature Store.

- The same script also runs hourly (automated via GitHub Actions) to fetch real-time updates and append new data to the feature store.

- This ensures that the model always has access to the latest information.

### 3. Model Training

- The model training is handled by the script AQI_Model_training_v2.py.

- Data is pulled from the Hopsworks Feature Store and used to train a Random Forest Regressor model.

- The model predicts AQI levels for the next 72 hours.

- Trained models are stored both locally (aqi_model_push/) and in the Hopsworks Model Registry.

- The training process is scheduled to run automatically every 12 hours using GitHub Actions.

### 4. Deployment and Visualization

- The trained model is deployed and visualized through a Streamlit web app (app_streamlit_aqi_v6.py).

- The dashboard shows:

-- Current AQI values

-- Forecasted AQI values for the next 72 hours

-- Graphical trends for better interpretability

### 5. Continuous Automation

- The pipeline is fully automated using GitHub Actions, which handle:

- Hourly data ingestion

- Periodic model retraining

This setup enables the project to remain serverless and maintenance-free.



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

<img width="1280" height="367" alt="image" src="https://github.com/user-attachments/assets/1078d8f2-b781-4bba-a7bc-4a976d525067" />
<img width="1280" height="557" alt="image" src="https://github.com/user-attachments/assets/9df388a8-a25e-49d8-9115-7b1cc112160b" />
<img width="1280" height="500" alt="image" src="https://github.com/user-attachments/assets/0fc26742-a770-41db-84dd-b1c7562c31c7" />
<img width="1280" height="434" alt="image" src="https://github.com/user-attachments/assets/3374ad1d-8320-4e3b-b75c-3ef8ec4dbf8c" />
<img width="1280" height="540" alt="image" src="https://github.com/user-attachments/assets/660d2967-7aa7-4862-b692-573a30974b61" />
<img width="1280" height="468" alt="image" src="https://github.com/user-attachments/assets/d7373e79-07e5-4b13-a2d6-db49a9dd2ac7" />






