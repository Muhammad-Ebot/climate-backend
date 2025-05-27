import pandas as pd
import numpy as np
import joblib
import requests
import logging
import os
from datetime import datetime, timedelta
from geopy.distance import geodesic
import warnings
from pandas.errors import PerformanceWarning
from django.conf import settings

# Suppress specific pandas warnings
warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrame.fillna with 'method' is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Downcasting object dtype arrays on .fillna.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Trying to unpickle estimator.*")

# Set up logging
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

logger.info(f"Looking for rainfall models in: {MODEL_DIR}")

# Province model and scaler mappings
PROVINCE_MODEL_MAP = {
    "sindh": "predictor/rainfall/models/sindh_rainfall_prediction_model.pkl",
    "punjab": "predictor/rainfall/models/pun_rainfall_prediction_model.pkl",
    "az": "predictor/rainfall/models/az_rainfall_prediction_model.pkl",
    "balochistan": "predictor/rainfall/models/balo_rainfall_prediction_model.pkl"
}

PROVINCE_SCALER_MAP = {
    "sindh": "predictor/rainfall/models/sindh_rainfall_scaler.pkl",
    "punjab": "predictor/rainfall/models/pun_rainfall_scaler.pkl",
    "az": "predictor/rainfall/models/az_rainfall_scaler.pkl",
    "balochistan": "predictor/rainfall/models/balo_rainfall_scaler.pkl"
}

# Pakistan's overall geographical boundaries
PAKISTAN_BOUNDS = {
    "lat_min": 23.5,
    "lat_max": 37.1,
    "lon_min": 60.9,
    "lon_max": 77.9
}

PROVINCE_BOUNDS = {
    'sindh': {'lat_min': 23, 'lat_max': 28.5, 'lon_min': 66, 'lon_max': 71},
    'punjab': {'lat_min': 28, 'lat_max': 34, 'lon_min': 69, 'lon_max': 75},
    'balochistan': {'lat_min': 24, 'lat_max': 31.5, 'lon_min': 60.5, 'lon_max': 69},
    'az': {'lat_min': 32, 'lat_max': 36.5, 'lon_min': 69, 'lon_max': 74}
}

RAIN_CATEGORIES = {
    0: "No rain",
    1: "Weak rain",
    2: "Moderate rain",
    3: "Heavy rain",
    4: "Severe rain"
}

# Global variables for models and scalers
LOADED_MODELS = {}
LOADED_SCALERS = {}


def get_weather_params_for_province(lat, lon, start_date, target_date, province):
    """
    Get the appropriate weather parameters based on the province.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        start_date (datetime): Start date for data
        target_date (datetime): Target date for prediction
        province (str): Province name ('sindh', 'punjab', etc.)
        
    Returns:
        dict: Parameters for weather API request
    """
    base_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Karachi",
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": target_date.strftime('%Y-%m-%d')
    }
    
    if province.lower() == 'sindh':
        base_params["hourly"] = [
            'relative_humidity_2m',
            'dew_point_2m',
            'rain',
            'surface_pressure',
            'cloud_cover_low',
            'cloud_cover_mid',
            'wind_gusts_10m',
            'soil_moisture_0_to_7cm',
            'soil_moisture_28_to_100cm'
        ]
    else:
        base_params["hourly"] = [
            'relative_humidity_2m',
            'dew_point_2m',
            'rain',
            'surface_pressure',
            'cloud_cover_low',
            'cloud_cover_mid',
            'cloud_cover_high',
            'wind_gusts_10m',
            'soil_moisture_0_to_7cm',
            'soil_moisture_28_to_100cm'
        ]
    
    return base_params


def create_dataframe_from_response(data, province):
    """
    Create DataFrame from API response based on province.
    
    Args:
        data (dict): API response data
        province (str): Province name
        
    Returns:
        DataFrame: Processed weather data
    """
    if province.lower() == 'sindh':
        df = pd.DataFrame({
            'date': pd.to_datetime(data['hourly']['time']),
            'relative_humidity_2m': data['hourly']['relative_humidity_2m'],
            'dew_point_2m': data['hourly']['dew_point_2m'],
            'rain': data['hourly']['rain'],
            'surface_pressure': data['hourly']['surface_pressure'],
            'cloud_cover_low': data['hourly']['cloud_cover_low'],
            'cloud_cover_mid': data['hourly']['cloud_cover_mid'],
            'wind_gusts_10m': data['hourly']['wind_gusts_10m'],
            'soil_moisture_0_to_7cm': data['hourly']['soil_moisture_0_to_7cm'],
            'soil_moisture_28_to_100cm': data['hourly']['soil_moisture_28_to_100cm']
        })
    else:
        df = pd.DataFrame({
            'date': pd.to_datetime(data['hourly']['time']),
            'relative_humidity_2m': data['hourly']['relative_humidity_2m'],
            'dew_point_2m': data['hourly']['dew_point_2m'],
            'rain': data['hourly']['rain'],
            'surface_pressure': data['hourly']['surface_pressure'],
            'cloud_cover_low': data['hourly']['cloud_cover_low'],
            'cloud_cover_mid': data['hourly']['cloud_cover_mid'],
            'cloud_cover_high': data['hourly']['cloud_cover_high'],
            'wind_gusts_10m': data['hourly']['wind_gusts_10m'],
            'soil_moisture_0_to_7cm': data['hourly']['soil_moisture_0_to_7cm'],
            'soil_moisture_28_to_100cm': data['hourly']['soil_moisture_28_to_100cm']
        })
    
    # Handle missing values in the raw data
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df


def is_coordinates_in_pakistan(lat, lon):
    """
    Check if coordinates fall within Pakistan's geographical boundaries.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        bool: True if coordinates are within Pakistan's boundaries
    """
    return (PAKISTAN_BOUNDS['lat_min'] <= lat <= PAKISTAN_BOUNDS['lat_max'] and 
            PAKISTAN_BOUNDS['lon_min'] <= lon <= PAKISTAN_BOUNDS['lon_max'])


def get_province_from_coordinates(lat, lon):
    """
    Determine province based on latitude and longitude coordinates.
    First checks if coordinates are within Pakistan, then determines the province.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        str or None: Province name or None if coordinates don't match any province
    """
    # First check if coordinates are within Pakistan
    if not is_coordinates_in_pakistan(lat, lon):
        return None
        
    for province, bounds in PROVINCE_BOUNDS.items():
        if (bounds['lat_min'] <= lat <= bounds['lat_max'] and 
            bounds['lon_min'] <= lon <= bounds['lon_max']):
            return province
    return None


def load_model_and_scaler(province):
    """
    Load model and scaler for a specific province.
    
    Args:
        province (str): Province name
        
    Returns:
        tuple: (model, scaler) or (None, None) if loading fails
    """
    if province in LOADED_MODELS and province in LOADED_SCALERS:
        return LOADED_MODELS[province], LOADED_SCALERS[province]
    
    try:
        model_path = PROVINCE_MODEL_MAP.get(province)
        scaler_path = PROVINCE_SCALER_MAP.get(province)
        
        if not model_path or not scaler_path:
            logger.error(f"No model/scaler path defined for province: {province}")
            return None, None
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None, None
            
        if not os.path.exists(scaler_path):
            logger.error(f"Scaler file not found: {scaler_path}")
            return None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        LOADED_MODELS[province] = model
        LOADED_SCALERS[province] = scaler
        
        logger.info(f"Successfully loaded model and scaler for {province}")
        return model, scaler
        
    except Exception as e:
        logger.error(f"Error loading model/scaler for {province}: {str(e)}")
        return None, None


def fetch_forecast(lat, lon, prediction_date=None):
    """
    Fetch weather data from Open-Meteo API (forecast or historical).

    Args:
        lat (float): Latitude
        lon (float): Longitude
        prediction_date (str or datetime, optional): Date to predict for (default: today)

    Returns:
        tuple: (DataFrame with hourly weather data, target_date as datetime)
    """
    try:
        if prediction_date:
            target_date = pd.to_datetime(prediction_date)
        else:
            target_date = pd.Timestamp.now().normalize()

        # Calculate start date (10 days before target date for lag features)
        start_date = target_date - pd.Timedelta(days=10)

        today = pd.Timestamp.now().normalize()
        max_forecast_date = today + pd.Timedelta(days=16)
        min_historical_date = today - pd.Timedelta(days=5)

        # Determine whether to use forecast or historical API
        if target_date >= today and target_date <= max_forecast_date:
            api_type = 'forecast'
            url = "https://api.open-meteo.com/v1/forecast"
        elif target_date <= min_historical_date:
            api_type = 'historical'
            url = "https://archive-api.open-meteo.com/v1/archive"
        else:
            raise ValueError(
                f"Requested date {target_date.strftime('%Y-%m-%d')} is not available. "
                f"Use historical API for dates before {min_historical_date.strftime('%Y-%m-%d')} "
                f"or forecast API for dates between {today.strftime('%Y-%m-%d')} and {max_forecast_date.strftime('%Y-%m-%d')}"
            )

        # Get province to determine which parameters to request
        province = get_province_from_coordinates(lat, lon)
        if not province:
            raise ValueError("Coordinates are not within any supported province")

        logger.info(f"Using {api_type.upper()} API for lat={lat}, lon={lon}, "
                   f"date range={start_date.strftime('%Y-%m-%d')} to {target_date.strftime('%Y-%m-%d')}")

        params = get_weather_params_for_province(
            lat=lat,
            lon=lon,
            start_date=start_date,
            target_date=target_date,
            province=province
        )

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        
        # Check if we have the expected data structure
        if 'hourly' not in data or 'time' not in data['hourly']:
            raise ValueError("Invalid response format from weather API")

        df = create_dataframe_from_response(data, province)
        logger.info(f"Successfully fetched {len(df)} hours of weather data")
        return df, target_date

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching weather data: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        raise


def _engineer_sindh_features(df):
    """Feature engineering specific to Sindh province"""
    df_feat = df.copy()

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_feat['date']):
        df_feat['date'] = pd.to_datetime(df_feat['date'])

    # Sort by date for accurate lagging
    df_feat = df_feat.sort_values('date').reset_index(drop=True)

    # --- Cyclical Time Features ---
    df_feat['hour'] = df_feat['date'].dt.hour
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)

    df_feat['day_of_year'] = df_feat['date'].dt.dayofyear
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['day_of_year'] / 365.25)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['day_of_year'] / 365.25)

    df_feat['month'] = df_feat['date'].dt.month
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)

    df_feat['year'] = df_feat['date'].dt.year

    # --- Seasonal Indicators ---
    df_feat['monsoon'] = df_feat['month'].between(6, 9).astype(int)
    df_feat['pre_monsoon'] = df_feat['month'].between(4, 5).astype(int)
    df_feat['post_monsoon'] = df_feat['month'].between(10, 11).astype(int)

    # --- Lag Features ---
    lag_features = [
        'relative_humidity_2m', 'dew_point_2m', 'surface_pressure',
        'cloud_cover_low', 'cloud_cover_mid', 'wind_gusts_10m',
        'soil_moisture_0_to_7cm', 'soil_moisture_28_to_100cm'
    ]
    lag_hours = [1, 2, 3, 6, 12, 24]

    for feature in lag_features:
        for lag in lag_hours:
            lag_col = f'{feature}_lag_{lag}h'
            df_feat[lag_col] = df_feat[feature].shift(lag)

    # --- Rolling Features ---
    rolling_windows = [3, 6, 12, 24, 48]

    for feature in lag_features:
        for window in rolling_windows:
            df_feat[f'{feature}_rollmean_{window}h'] = df_feat[feature].rolling(window=window, min_periods=1).mean()
            df_feat[f'{feature}_rollstd_{window}h'] = df_feat[feature].rolling(window=window, min_periods=1).std()
            df_feat[f'{feature}_rollmin_{window}h'] = df_feat[feature].rolling(window=window, min_periods=1).min()
            df_feat[f'{feature}_rollmax_{window}h'] = df_feat[feature].rolling(window=window, min_periods=1).max()

    # --- Combined Features ---
    df_feat['cloud_cover_total'] = df_feat['cloud_cover_low'] + df_feat['cloud_cover_mid']
    df_feat['soil_moisture_gradient'] = df_feat['soil_moisture_0_to_7cm'] - df_feat['soil_moisture_28_to_100cm']
    df_feat['humid_dew_diff'] = df_feat['relative_humidity_2m'] - df_feat['dew_point_2m']

    # --- Pressure Changes ---
    df_feat['pressure_change_1h'] = df_feat['surface_pressure'] - df_feat['surface_pressure_lag_1h']
    df_feat['pressure_change_3h'] = df_feat['surface_pressure'] - df_feat['surface_pressure_lag_3h']
    df_feat['pressure_change_6h'] = df_feat['surface_pressure'] - df_feat['surface_pressure_lag_6h']

    # --- Wind Features ---
    # Avoid division by zero
    df_feat['wind_pressure_index'] = np.where(
        df_feat['surface_pressure'] != 0,
        df_feat['wind_gusts_10m'] / df_feat['surface_pressure'],
        0
    )
    df_feat['wind_change_1h'] = df_feat['wind_gusts_10m'] - df_feat['wind_gusts_10m_lag_1h']
    df_feat['wind_change_3h'] = df_feat['wind_gusts_10m'] - df_feat['wind_gusts_10m_lag_3h']

    # --- Cloud Features ---
    df_feat['cloud_change_1h'] = (df_feat['cloud_cover_total'] - 
                                 (df_feat['cloud_cover_low_lag_1h'] + df_feat['cloud_cover_mid_lag_1h']))
    df_feat['cloud_change_3h'] = (df_feat['cloud_cover_total'] - 
                                 (df_feat['cloud_cover_low_lag_3h'] + df_feat['cloud_cover_mid_lag_3h']))

    # Fill remaining NaNs
    df_feat = df_feat.fillna(method='ffill').fillna(method='bfill').fillna(0)

    return df_feat


def _engineer_other_province_features(df, latitude=None, longitude=None):
    """Feature engineering for non-Sindh provinces"""
    df_feat = df.copy()

    # Convert date if needed
    if not pd.api.types.is_datetime64_any_dtype(df_feat['date']):
        df_feat['date'] = pd.to_datetime(df_feat['date'])

    # Add location features
    if latitude is not None and longitude is not None:
        df_feat['lat'] = float(latitude)
        df_feat['lon'] = float(longitude)
    else:
        logger.warning("Latitude and longitude not provided. Using default values.")
        df_feat['lat'] = 0.0
        df_feat['lon'] = 0.0

    # Extract cyclical time features
    df_feat['hour'] = df_feat['date'].dt.hour
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour']/24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour']/24)

    df_feat['day_of_year'] = df_feat['date'].dt.dayofyear
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['day_of_year']/365.25)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['day_of_year']/365.25)

    df_feat['month'] = df_feat['date'].dt.month
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month']/12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month']/12)

    df_feat['year'] = df_feat['date'].dt.year

    # Seasonal indicators
    df_feat['monsoon'] = df_feat['month'].between(6, 9).astype(int)
    df_feat['pre_monsoon'] = df_feat['month'].between(4, 5).astype(int)
    df_feat['post_monsoon'] = df_feat['month'].between(10, 11).astype(int)

    # Sort by date for lag/rolling features
    df_feat.sort_values('date', inplace=True)

    # Create lagged features
    lag_features = ['relative_humidity_2m', 'dew_point_2m', 'surface_pressure',
                   'cloud_cover_low', 'cloud_cover_mid', 'wind_gusts_10m',
                   'soil_moisture_0_to_7cm', 'soil_moisture_28_to_100cm']
    
    if 'cloud_cover_high' in df_feat.columns:
        lag_features.append('cloud_cover_high')

    lag_hours = [1, 2, 3, 6, 12, 24]

    for feature in lag_features:
        for lag in lag_hours:
            lag_col = f'{feature}_lag_{lag}h'
            df_feat[lag_col] = df_feat[feature].shift(lag)

    # Create rolling window features
    rolling_windows = [3, 6, 12, 24, 48]

    for feature in lag_features:
        for window in rolling_windows:
            df_feat[f'{feature}_rollmean_{window}h'] = df_feat[feature].rolling(window=window).mean()
            df_feat[f'{feature}_rollstd_{window}h'] = df_feat[feature].rolling(window=window).std()
            df_feat[f'{feature}_rollmin_{window}h'] = df_feat[feature].rolling(window=window).min()
            df_feat[f'{feature}_rollmax_{window}h'] = df_feat[feature].rolling(window=window).max()

    # Create combined features
    df_feat['cloud_cover_total'] = df_feat['cloud_cover_low'] + df_feat['cloud_cover_mid']
    
    if 'cloud_cover_high' in df_feat.columns:
        df_feat['cloud_cover_total'] += df_feat['cloud_cover_high']

    # Meteorological derived features
    df_feat['soil_moisture_gradient'] = df_feat['soil_moisture_0_to_7cm'] - df_feat['soil_moisture_28_to_100cm']
    df_feat['humid_dew_diff'] = df_feat['relative_humidity_2m'] - df_feat['dew_point_2m']
    df_feat['pressure_change_1h'] = df_feat['surface_pressure'] - df_feat['surface_pressure_lag_1h']
    df_feat['pressure_change_3h'] = df_feat['surface_pressure'] - df_feat['surface_pressure_lag_3h']
    df_feat['pressure_change_6h'] = df_feat['surface_pressure'] - df_feat['surface_pressure_lag_6h']
    df_feat['wind_pressure_index'] = df_feat['wind_gusts_10m'] / df_feat['surface_pressure']
    df_feat['wind_change_1h'] = df_feat['wind_gusts_10m'] - df_feat['wind_gusts_10m_lag_1h']
    df_feat['wind_change_3h'] = df_feat['wind_gusts_10m'] - df_feat['wind_gusts_10m_lag_3h']
    df_feat['cloud_change_1h'] = df_feat['cloud_cover_total'] - (df_feat['cloud_cover_low_lag_1h'] + df_feat['cloud_cover_mid_lag_1h'])
    df_feat['cloud_change_3h'] = df_feat['cloud_cover_total'] - (df_feat['cloud_cover_low_lag_3h'] + df_feat['cloud_cover_mid_lag_3h'])

    # Drop rows with NaN values
    df_feat = df_feat.dropna()

    return df_feat


def engineer_features_for_prediction(df, province, latitude=None, longitude=None):
    """
    Province-specific feature engineering.
    
    Args:
        df (DataFrame): Raw weather data
        province (str): Province name
        latitude (float): Optional latitude
        longitude (float): Optional longitude
        
    Returns:
        DataFrame: Engineered features
    """
    if province.lower() == 'sindh':
        return _engineer_sindh_features(df)
    else:
        return _engineer_other_province_features(df, latitude, longitude)


def validate_prediction_features(X, expected_features=None):
    """
    Validate and align prediction features with model expectations.
    
    Args:
        X (DataFrame): Feature matrix
        expected_features (list, optional): List of expected feature names
        
    Returns:
        DataFrame: Validated and aligned feature matrix
    """
    if expected_features is not None:
        # Ensure all expected features are present
        missing_features = set(expected_features) - set(X.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            for feature in missing_features:
                X[feature] = 0
        
        # Reorder columns to match expected order
        X = X[expected_features]
    
    # Final check for NaN values
    if X.isna().any().any():
        logger.warning("Found NaN values in prediction features, filling with 0")
        X = X.fillna(0)
    
    return X


def predict_rainfall_24h(latitude, longitude, prediction_date=None):
    """
    Make 24-hour rainfall predictions for a specific day.
    
    Args:
        latitude (float): Latitude for forecast
        longitude (float): Longitude for forecast
        prediction_date (str, optional): Date string to predict for (YYYY-MM-DD), defaults to today
        
    Returns:
        dict: Dictionary with predictions for 24 hours and summary information
    """
    try:
        # Validate coordinates format
        if not (-90 <= latitude <= 90):
            return {
                'error': True,
                'message': f'Invalid latitude: {latitude}. Must be between -90 and 90.',
                'prediction_date': pd.Timestamp.now().date() if prediction_date is None else pd.to_datetime(prediction_date).date()
            }
        
        if not (-180 <= longitude <= 180):
            return {
                'error': True,
                'message': f'Invalid longitude: {longitude}. Must be between -180 and 180.',
                'prediction_date': pd.Timestamp.now().date() if prediction_date is None else pd.to_datetime(prediction_date).date()
            }

        # Check if coordinates are within Pakistan's boundaries
        if not is_coordinates_in_pakistan(latitude, longitude):
            return {
                'error': True,
                'message': f'Coordinates ({latitude}, {longitude}) are outside Pakistan\'s geographical boundaries. '
                          f'Pakistan\'s boundaries are approximately: '
                          f'Latitude: {PAKISTAN_BOUNDS["lat_min"]}° to {PAKISTAN_BOUNDS["lat_max"]}°N, '
                          f'Longitude: {PAKISTAN_BOUNDS["lon_min"]}° to {PAKISTAN_BOUNDS["lon_max"]}°E',
                'prediction_date': pd.Timestamp.now().date() if prediction_date is None else pd.to_datetime(prediction_date).date(),
                'pakistan_bounds': PAKISTAN_BOUNDS
            }

        # 1. Determine province and load appropriate model
        province = get_province_from_coordinates(latitude, longitude)
        if not province:
            return {
                'error': True,
                'message': f'Coordinates ({latitude}, {longitude}) are within Pakistan but outside supported provinces. '
                          f'Supported provinces: {", ".join(PROVINCE_BOUNDS.keys())}. '
                          f'This location might be in an unsupported region like Gilgit-Baltistan, KPK, or disputed territories.',
                'prediction_date': pd.Timestamp.now().date() if prediction_date is None else pd.to_datetime(prediction_date).date(),
                'supported_provinces': list(PROVINCE_BOUNDS.keys())
            }

        model, scaler = load_model_and_scaler(province)
        if model is None or scaler is None:
            return {
                'error': True,
                'message': f'Rainfall prediction model is not available for {province} province.',
                'prediction_date': pd.Timestamp.now().date() if prediction_date is None else pd.to_datetime(prediction_date).date()
            }

        # 2. Fetch forecast data
        df, target_date = fetch_forecast(latitude, longitude, prediction_date=prediction_date)

        # 3. Engineer features
        logger.info(f"Engineering features for {len(df)} rows")
        features_df = engineer_features_for_prediction(df, province, latitude, longitude)

        # 4. Filter to target date hours
        target_day = target_date.date()
        pred_features_df = features_df[features_df['date'].dt.date == target_day].copy()
        logger.info(f"Filtered to {len(pred_features_df)} hours for prediction date {target_day}")

        if len(pred_features_df) == 0:
            return {
                'error': True,
                'message': f'No data available for prediction date {target_day}',
                'prediction_date': target_day
            }

        if len(pred_features_df) != 24:
            logger.warning(f"Expected 24 hours but got {len(pred_features_df)} hours for {target_day}")

        # 5. Prepare feature matrix
        excluded_cols = ['date', 'rain', 'rain_category']
        feature_cols = [col for col in pred_features_df.columns if col not in excluded_cols]
        X = pred_features_df[feature_cols].copy()

        # 6. Validate and scale features
        X = validate_prediction_features(X)
        
        # Get feature names that the scaler expects
        try:
            expected_features = scaler.feature_names_in_
            X = validate_prediction_features(X, expected_features)
        except AttributeError:
            # Scaler doesn't have feature_names_in_ attribute
            pass

        X_scaled = scaler.transform(X)

        # 7. Make predictions
        predictions = model.predict(X_scaled)
        probas = model.predict_proba(X_scaled)

        # 8. Prepare results
        results_df = pred_features_df[['date', 'hour', 'relative_humidity_2m', 
                                      'cloud_cover_mid', 'wind_gusts_10m', 
                                      'soil_moisture_0_to_7cm']].copy()
        if 'rain' in pred_features_df.columns:
            results_df['rain'] = pred_features_df['rain']
        
        results_df['rain_category'] = predictions
        results_df['prediction_confidence'] = np.max(probas, axis=1)
        results_df['rain_category_label'] = results_df['rain_category'].map(RAIN_CATEGORIES)

        # Sort by time
        results_df = results_df.sort_values('date').reset_index(drop=True)

        # 9. Create summary
        summary = results_df.groupby('rain_category_label').size().reset_index()
        summary.columns = ['rain_category', 'hours_count']
        summary['percentage'] = (summary['hours_count'] / len(results_df) * 100).round(1)
        summary_dict = summary.set_index('rain_category').to_dict('index')

        # 10. Convert to serializable format
        hourly_predictions = []
        for _, row in results_df.iterrows():
            prediction = {
                'date': row['date'].isoformat(),
                'hour': int(row['hour']),
                'rain_category': int(row['rain_category']),
                'rain_category_label': row['rain_category_label'],
                'relative_humidity_2m': float(row['relative_humidity_2m']),
                'cloud_cover_mid': float(row['cloud_cover_mid']),
                'wind_gusts_10m': float(row['wind_gusts_10m']),
                'soil_moisture_0_to_7cm': float(row['soil_moisture_0_to_7cm']),
                'prediction_confidence': float(row['prediction_confidence'])
            }

            if 'rain' in results_df.columns:
                prediction['actual_rain'] = float(row['rain']) if not pd.isna(row['rain']) else 0.0
            hourly_predictions.append(prediction)

        return {
            'error': False,
            'province': province,
            'coordinates': {'latitude': latitude, 'longitude': longitude},
            'prediction_date': target_day,
            'total_hours': len(hourly_predictions),
            'hourly_predictions': hourly_predictions,
            'summary': summary_dict,
            'model_info': {
                'province': province,
                'total_predictions': len(predictions)
            }
        }

    except Exception as e:
        logger.error(f"Rainfall prediction failed: {str(e)}")
        logger.exception("Detailed traceback:")
        return {
            'error': True,
            'message': str(e),
            'prediction_date': pd.Timestamp.now().date() if prediction_date is None else pd.to_datetime(prediction_date).date()
        }


def get_pakistan_bounds():
    """
    Get Pakistan's geographical boundaries.
    
    Returns:
        dict: Dictionary with Pakistan's lat/lon boundaries
    """
    return PAKISTAN_BOUNDS.copy()


def get_location_info(lat, lon):
    """
    Get comprehensive location information for given coordinates.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        dict: Location information including Pakistan status and province
    """
    info = {
        'coordinates': {'latitude': lat, 'longitude': lon},
        'is_in_pakistan': is_coordinates_in_pakistan(lat, lon),
        'province': None,
        'supported_for_prediction': False
    }
    
    if info['is_in_pakistan']:
        province = get_province_from_coordinates(lat, lon)
        if province:
            info['province'] = province
            info['supported_for_prediction'] = True
        else:
            info['message'] = 'Location is in Pakistan but outside supported provinces'
    else:
        info['message'] = 'Location is outside Pakistan boundaries'
    
    return info


def validate_coordinates_for_province(lat, lon, province):
    """
    Check if coordinates are within the specified province bounds.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude  
        province (str): Province name
        
    Returns:
        bool: True if coordinates are within province bounds
    """
    if province not in PROVINCE_BOUNDS:
        return False
    
    bounds = PROVINCE_BOUNDS[province]
    return (bounds['lat_min'] <= lat <= bounds['lat_max'] and 
            bounds['lon_min'] <= lon <= bounds['lon_max'])


def get_available_provinces():
    """
    Get list of available provinces for prediction.
    
    Returns:
        list: List of province names
    """
    return list(PROVINCE_BOUNDS.keys())

