# heatwave/pipeline.py

# heatwave/pipeline.py

import pandas as pd
import joblib
import requests
import os
from django.conf import settings

province_model_map = {
    "sindh": "heatwave_sindh_XGB.pkl",
    "punjab": "heatwave_punjab_XGB.pkl",
    "kpk": "heatwave_kpk_XGB.pkl",
    "balochistan": "heatwave_balochistan_XGB.pkl"
}

PROVINCE_BOUNDS = {
    'sindh': {'lat_min': 23, 'lat_max': 28.5, 'lon_min': 66, 'lon_max': 71},
    'punjab': {'lat_min': 28, 'lat_max': 34, 'lon_min': 69, 'lon_max': 75},
    'balochistan': {'lat_min': 24, 'lat_max': 31.5, 'lon_min': 60.5, 'lon_max': 69},
    'kpk': {'lat_min': 32, 'lat_max': 36.5, 'lon_min': 69, 'lon_max': 74}
}


def get_province_from_coordinates(lat, lon):
    for province, bounds in PROVINCE_BOUNDS.items():
        if bounds['lat_min'] <= lat <= bounds['lat_max'] and bounds['lon_min'] <= lon <= bounds['lon_max']:
            return province
    return None


def fetch_weather_data(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration",
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


def predict_heatwave(latitude, longitude):
    try:
        df = fetch_weather_data(latitude, longitude)
    except Exception as e:
        return {"error": str(e)}

    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map({
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall", 10: "fall", 11: "fall"
    })
    df["temp_7d_avg"] = df["temperature_2m_mean"].rolling(window=7, min_periods=1).mean()
    df["temp_14d_avg"] = df["temperature_2m_mean"].rolling(window=14, min_periods=1).mean()
    df["heatwave_threshold"] = 35
    df = pd.concat([df, pd.get_dummies(df["season"], prefix="season")], axis=1)

    expected_features = [
        "date", "temperature_2m_mean", "wind_speed_10m_max", "wind_direction_10m_dominant",
        "shortwave_radiation_sum", "et0_fao_evapotranspiration", "day_of_year",
        "season", "month", "precipitation_sum", "temp_7d_avg", "temp_14d_avg",
        "heatwave_threshold", "season_summer"
    ]

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]

    province = get_province_from_coordinates(latitude, longitude)
    model_filename = province_model_map.get(province)

    if not model_filename:
        return {"error": f"No model available for province: {province}"}

    model_path = os.path.join(settings.MODEL_DIR, model_filename)
    if not os.path.exists(model_path):
        return {"error": f"Model file not found: {model_path}"}

    model = joblib.load(model_path)
    print(f"Making prediction using model: {model_filename} for province: {province}")

    df["heatwave_prediction"] = model.predict(df)

    return df[["date", "temperature_2m_mean", "heatwave_prediction"]].to_dict(orient="records")
