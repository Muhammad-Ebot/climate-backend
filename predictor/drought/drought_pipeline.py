import pandas as pd
import numpy as np
import requests
import joblib
import xgboost as xgb
from geopy.distance import geodesic
import os

# Base paths for region-specific models and bounds
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DATA = {
    'sindh': {
        'grid_csv': os.path.join(BASE_DIR, 'models/sindh_grids.csv'),
        'model_json': os.path.join(BASE_DIR, 'models/drought_model_sindh.json'),
        'components': os.path.join(BASE_DIR, 'models/drought_components_sindh.joblib'),
        'bounds': {'lat_min': 23, 'lat_max': 28.5, 'lon_min': 66, 'lon_max': 71}
    },
    'punjab': {
        'grid_csv': os.path.join(BASE_DIR, 'models/lahore_grids.csv'),
        'model_json': os.path.join(BASE_DIR, 'models/drought_model_lahore.json'),
        'components': os.path.join(BASE_DIR, 'models/drought_components_lahore.joblib'),
        'bounds': {'lat_min': 28, 'lat_max': 34, 'lon_min': 69, 'lon_max': 75}
    },
    'balochistan': {
        'grid_csv': os.path.join(BASE_DIR, 'models/quetta_grids.csv'),
        'model_json': os.path.join(BASE_DIR, 'models/drought_model_quetta.json'),
        'components': os.path.join(BASE_DIR, 'models/drought_components_quetta.joblib'),
        'bounds': {'lat_min': 24, 'lat_max': 31.5, 'lon_min': 60.5, 'lon_max': 69}
    },
    'kpk': {
        'grid_csv': os.path.join(BASE_DIR, 'models/peshawer_grids.csv'),
        'model_json': os.path.join(BASE_DIR, 'models/drought_model_peshawer.json'),
        'components': os.path.join(BASE_DIR, 'models/drought_components_peshawer.joblib'),
        'bounds': {'lat_min': 32, 'lat_max': 36.5, 'lon_min': 69, 'lon_max': 74}
    }
}

# Identify which region the coordinates fall into
def detect_region(lat, lon):
    for region, info in MODEL_DATA.items():
        b = info['bounds']
        if b['lat_min'] <= lat <= b['lat_max'] and b['lon_min'] <= lon <= b['lon_max']:
            return region
    return 'sindh'  # fallback

# Find nearest grid cell using geodesic distance
def find_nearest_grid(lat, lon, grid_df):
    min_distance = float('inf')
    nearest_grid_id = None
    for _, row in grid_df.iterrows():
        distance = geodesic((lat, lon), (row['lat'], row['lon'])).km
        if distance < min_distance:
            min_distance = distance
            nearest_grid_id = row['grid_id']
    return int(nearest_grid_id)

# Fetch 15-day forecast from Open-Meteo API with API key
def fetch_weather_data(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration"
        ],
        "timezone": "Asia/Karachi",
        "forecast_days": 15,
        "timezone": "auto",
        "apikey": "hNaRTVY4v6i4jhWK"
    }

    response = requests.get("https://customer-api.open-meteo.com/v1/forecast", params=params)

    if response.status_code != 200:
        raise Exception("Weather data fetch failed.")

    data = response.json()
    return pd.DataFrame({
        "date": data["daily"]["time"],
        "temperature_2m_max": data["daily"]["temperature_2m_max"],
        "temperature_2m_min": data["daily"]["temperature_2m_min"],
        "temperature_2m_mean": data["daily"]["temperature_2m_mean"],
        "precipitation_sum": data["daily"]["precipitation_sum"],
        "wind_speed_10m_max": data["daily"]["wind_speed_10m_max"],
        "wind_gusts_10m_max": data["daily"]["wind_gusts_10m_max"],
        "wind_direction_10m_dominant": data["daily"]["wind_direction_10m_dominant"],
        "shortwave_radiation_sum": data["daily"]["shortwave_radiation_sum"],
        "et0_fao_evapotranspiration": data["daily"]["et0_fao_evapotranspiration"]
    })

# Predict drought category using pre-trained model and features
def predict_drought(forecast_df, grid_id, model_path, components_path):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    components = joblib.load(components_path)
    preprocessor = components['preprocessor']
    engineered_features = components['engineered_features']

    forecast_df['grid_id'] = grid_id
    forecast_df['month'] = pd.to_datetime(forecast_df['date']).dt.month

    # Generate engineered features if missing
    for feature in engineered_features:
        if feature not in forecast_df.columns:
            if feature == 'season_encoded':
                forecast_df['season_encoded'] = forecast_df['month'].map(lambda m: 0 if m in [12, 1, 2] else 1 if m in [3, 4, 5] else 2 if m in [6, 7, 8] else 3)
            elif feature == 'temperature_range':
                forecast_df['temperature_range'] = forecast_df['temperature_2m_max'] - forecast_df['temperature_2m_min']
            elif feature == 'wind_direction_sin':
                forecast_df['wind_direction_sin'] = np.sin(np.radians(forecast_df['wind_direction_10m_dominant']))
            elif feature == 'wind_direction_cos':
                forecast_df['wind_direction_cos'] = np.cos(np.radians(forecast_df['wind_direction_10m_dominant']))
            elif feature in ['precipitation_30d_sum', 'precipitation_60d_sum', 'precipitation_90d_sum']:
                forecast_df[feature] = forecast_df['precipitation_sum'] * 30
            elif feature == 'dry_days_30d':
                forecast_df['dry_days_30d'] = (forecast_df['precipitation_sum'] < 1.0).astype(int) * 30
            elif feature == 'aridity_index':
                forecast_df['aridity_index'] = forecast_df['et0_fao_evapotranspiration'] / forecast_df['precipitation_sum'].replace(0, 0.1)
            elif feature == 'water_balance':
                forecast_df['water_balance'] = forecast_df['precipitation_sum'] - forecast_df['et0_fao_evapotranspiration']
            elif feature == 'water_balance_30d':
                forecast_df['water_balance_30d'] = (forecast_df['precipitation_sum'] - forecast_df['et0_fao_evapotranspiration']) * 30
            else:
                forecast_df[feature] = 0

    forecast_df.fillna(0, inplace=True)
    X = forecast_df[engineered_features]
    X_processed = preprocessor.transform(X)
    preds = model.predict(X_processed)
    probs = model.predict_proba(X_processed)

    drought_categories = {
        0: "Extreme Drought",
        1: "Severely Dry",
        2: "Moderately Dry",
        3: "Mild Drought",
        4: "Moderately Wet",
        5: "Very Wet",
        6: "Extremely Wet"
    }

    # Format predictions for output
    return [{
        'date': forecast_df.loc[i, 'date'],
        'grid_id': grid_id,
        'drought_level_numeric': int(preds[i]),
        'drought_category': drought_categories.get(preds[i], f"Level {preds[i]}"),
        'confidence': float(max(probs[i])),
        'class_probabilities': {
            drought_categories.get(cls, f"Level {cls}"): float(probs[i][j])
            for j, cls in enumerate(model.classes_)
        }
    } for i in range(len(preds))]

# 🔁 Main driver: detect region, fetch forecast, predict
def run_drought_forecast(lat_input, lon_input):
    region = detect_region(lat_input, lon_input)
    config = MODEL_DATA[region]
    grid_df = pd.read_csv(config['grid_csv'])
    grid_df.rename(columns=lambda x: x.strip().lower(), inplace=True)

    if 'grid_id' not in grid_df.columns:
        grid_df.rename(columns={'Grid_ID': 'grid_id'}, inplace=True)

    nearest_grid_id = find_nearest_grid(lat_input, lon_input, grid_df)
    nearest_row = grid_df[grid_df['grid_id'] == nearest_grid_id].iloc[0]
    lat, lon = nearest_row['lat'], nearest_row['lon']

    forecast_df = fetch_weather_data(lat, lon)  # Updated to use new function
    prediction_results = predict_drought(forecast_df, nearest_grid_id, config['model_json'], config['components'])

    for res in prediction_results:
        print(f"{res['date']} | Grid: {res['grid_id']} | Drought Category: {res['drought_category']} | Confidence: {res['confidence']*100:.2f}%")

    return prediction_results