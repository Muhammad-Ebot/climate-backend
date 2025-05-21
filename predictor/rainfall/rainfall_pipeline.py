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

# Suppress scikit-learn version inconsistency warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Trying to unpickle estimator.*")

# Set up logging with Django's logging configuration
logger = logging.getLogger(__name__)

# Define Django-compatible paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# For debugging, print the path to check it
logger.info(f"Looking for rainfall models in: {MODEL_DIR}")

# Prepare file paths
MODEL_PATH = os.path.join(MODEL_DIR, 'rainfall_prediction_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'rainfall_scaler.pkl')
GRID_DATA_PATH = os.path.join(MODEL_DIR, 'karachi_grid_points.csv')

# Add file existence checks
if not os.path.exists(MODEL_PATH):
    logger.warning(f"WARNING: Rainfall model file not found at {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    logger.warning(f"WARNING: Rainfall scaler file not found at {SCALER_PATH}")
if not os.path.exists(GRID_DATA_PATH):
    logger.warning(f"WARNING: Rainfall grid data file not found at {GRID_DATA_PATH}")
    
# Load grid data once at module level for performance
try:
    grid_data = pd.read_csv(GRID_DATA_PATH)
    grid_data.rename(columns=lambda x: x.strip().lower(), inplace=True)
    if 'grid_id' not in grid_data.columns:
        grid_data.rename(columns={'Grid_ID': 'grid_id'}, inplace=True)
    logger.info(f"Successfully loaded rainfall grid data with {len(grid_data)} points")
except Exception as e:
    logger.error(f"Error loading rainfall grid data: {str(e)}")
    grid_data = None

# Load model and scaler once at module level for performance
try:
    MODEL = joblib.load(MODEL_PATH)
    SCALER = joblib.load(SCALER_PATH)
    logger.info("Successfully loaded rainfall prediction model and scaler")
except Exception as e:
    logger.error(f"Error loading rainfall model or scaler: {str(e)}")
    MODEL = None
    SCALER = None


def find_nearest_grid(lat, lon):
    """
    Find the nearest grid point to given coordinates

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Nearest grid ID or None if grid data unavailable
    """
    if grid_data is None:
        logger.error("Grid data not available. Cannot find nearest grid.")
        return None

    min_distance = float('inf')
    nearest_grid_id = None
    for _, row in grid_data.iterrows():
        distance = geodesic((lat, lon), (row['lat'], row['lon'])).km
        if distance < min_distance:
            min_distance = distance
            nearest_grid_id = row['grid_id']

    logger.info(f"Found nearest grid point {nearest_grid_id} at distance {min_distance:.2f} km")
    return int(nearest_grid_id)


def fetch_forecast(lat, lon, prediction_date=None):
    """
    Fetch weather data from Open-Meteo API (forecast or historical).

    Args:
        lat: Latitude
        lon: Longitude
        prediction_date: Date to predict for (default: today)

    Returns:
        DataFrame with hourly weather data for the prediction date and prior dates,
        and the target date as a datetime object
    """
    try:
        if prediction_date:
            target_date = pd.to_datetime(prediction_date)
        else:
            target_date = pd.Timestamp.now().normalize()

        # Calculate start date (10 days before target date)
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
            raise ValueError("Requested date is too recent for historical data and too old for forecast data (within last 5 days).")

        logger.info(f"Using {api_type.upper()} API for lat={lat}, lon={lon}, date range={start_date.strftime('%Y-%m-%d')} to {target_date.strftime('%Y-%m-%d')}")

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                'relative_humidity_2m',
                'dew_point_2m',
                'rain',
                'surface_pressure',
                'cloud_cover_low',
                'cloud_cover_mid',
                'wind_gusts_10m',
                'soil_moisture_0_to_7cm',
                'soil_moisture_28_to_100cm'
            ],
            "timezone": "Asia/Karachi",
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": target_date.strftime('%Y-%m-%d')
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

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

        return df, target_date

    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise


def engineer_features_for_prediction(df):
    """
    Replicate EXACTLY the feature engineering from training
    
    Args:
        df: DataFrame with raw weather data
        
    Returns:
        DataFrame with engineered features
    """
    df_feat = df.copy()

    # Convert date if needed
    if not pd.api.types.is_datetime64_any_dtype(df_feat['date']):
        df_feat['date'] = pd.to_datetime(df_feat['date'])

    # --- Cyclical Time Features ---
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

    # --- Seasonal Indicators ---
    df_feat['monsoon'] = df_feat['month'].between(6, 9).astype(int)
    df_feat['pre_monsoon'] = df_feat['month'].between(4, 5).astype(int)
    df_feat['post_monsoon'] = df_feat['month'].between(10, 11).astype(int)

    # --- Lag Features ---
    lag_features = ['relative_humidity_2m', 'dew_point_2m', 'surface_pressure',
                   'cloud_cover_low', 'cloud_cover_mid', 'wind_gusts_10m',
                   'soil_moisture_0_to_7cm', 'soil_moisture_28_to_100cm']

    lag_hours = [1, 2, 3, 6, 12, 24]

    # Sort by date first for accurate lagging
    df_feat = df_feat.sort_values('date')

    for feature in lag_features:
        for lag in lag_hours:
            lag_col = f'{feature}_lag_{lag}h'
            df_feat[lag_col] = df_feat[feature].shift(lag)

    # --- Rolling Features ---
    rolling_windows = [3, 6, 12, 24, 48]

    for feature in lag_features:
        for window in rolling_windows:
            df_feat[f'{feature}_rollmean_{window}h'] = df_feat[feature].rolling(window=window).mean()
            df_feat[f'{feature}_rollstd_{window}h'] = df_feat[feature].rolling(window=window).std()
            df_feat[f'{feature}_rollmin_{window}h'] = df_feat[feature].rolling(window=window).min()
            df_feat[f'{feature}_rollmax_{window}h'] = df_feat[feature].rolling(window=window).max()

    # --- Combined Features ---
    df_feat['cloud_cover_total'] = df_feat['cloud_cover_low'] + df_feat['cloud_cover_mid']
    df_feat['soil_moisture_gradient'] = df_feat['soil_moisture_0_to_7cm'] - df_feat['soil_moisture_28_to_100cm']
    df_feat['humid_dew_diff'] = df_feat['relative_humidity_2m'] - df_feat['dew_point_2m']

    # --- Pressure Changes ---
    df_feat['pressure_change_1h'] = df_feat['surface_pressure'] - df_feat['surface_pressure_lag_1h']
    df_feat['pressure_change_3h'] = df_feat['surface_pressure'] - df_feat['surface_pressure_lag_3h']
    df_feat['pressure_change_6h'] = df_feat['surface_pressure'] - df_feat['surface_pressure_lag_6h']

    # --- Wind Features ---
    df_feat['wind_pressure_index'] = df_feat['wind_gusts_10m'] / df_feat['surface_pressure']
    df_feat['wind_change_1h'] = df_feat['wind_gusts_10m'] - df_feat['wind_gusts_10m_lag_1h']
    df_feat['wind_change_3h'] = df_feat['wind_gusts_10m'] - df_feat['wind_gusts_10m_lag_3h']

    # --- Cloud Features ---
    df_feat['cloud_change_1h'] = df_feat['cloud_cover_total'] - (df_feat['cloud_cover_low_lag_1h'] + df_feat['cloud_cover_mid_lag_1h'])
    df_feat['cloud_change_3h'] = df_feat['cloud_cover_total'] - (df_feat['cloud_cover_low_lag_3h'] + df_feat['cloud_cover_mid_lag_3h'])

    # Preserving rain column if it exists
    if 'rain' in df.columns:
        df_feat['rain'] = df['rain']  # Preserve original rain values

    return df_feat


def predict_rainfall_24h(latitude, longitude, prediction_date=None):
    """
    Make 24-hour rainfall predictions for a specific day
    
    Django-friendly function to be called from views
    
    Args:
        latitude (float): Latitude for forecast
        longitude (float): Longitude for forecast
        prediction_date (str, optional): Date string to predict for (YYYY-MM-DD), defaults to today
        
    Returns:
        dict: Dictionary with predictions for 24 hours and summary information
    """
    try:
        # Check if model is loaded
        if MODEL is None or SCALER is None:
            logger.error("Model or scaler not loaded properly")
            return {
                'error': True,
                'message': 'Rainfall prediction model is not available',
                'prediction_date': pd.Timestamp.now().date() if prediction_date is None else pd.to_datetime(prediction_date).date()
            }

        # 1. Find nearest grid point
        nearest_grid_id = find_nearest_grid(latitude, longitude)
        if nearest_grid_id is not None and grid_data is not None:
            nearest_row = grid_data[grid_data['grid_id'] == nearest_grid_id].iloc[0]
            grid_lat, grid_lon = nearest_row['lat'], nearest_row['lon']
            logger.info(f"Using nearest grid: {nearest_grid_id} at ({grid_lat}, {grid_lon})")

            # Use the grid point coordinates instead of the original ones
            latitude, longitude = grid_lat, grid_lon
        else:
            logger.warning("Could not find nearest grid. Using original coordinates.")

        # 2. Fetch forecast data (with history for lag features)
        df, target_date = fetch_forecast(
            latitude,
            longitude,
            prediction_date=prediction_date,
        )

        # 3. Engineer features
        logger.info(f"Engineering features for {len(df)} rows")
        features_df = engineer_features_for_prediction(df)

        features_df = features_df.dropna().reset_index(drop=True)

        # 4. Filter to just the 24 hours of the target date
        target_day = target_date.date()
        pred_features_df = features_df[features_df['date'].dt.date == target_day]
        logger.info(f"Filtered to {len(pred_features_df)} hours for prediction date {target_day}")

        if pred_features_df.isnull().any().any():
            logger.warning("⚠️ Prediction DataFrame still contains NaNs! Predictions may be unreliable.")

        # Check if we have 24 hours
        if len(pred_features_df) != 24:
            logger.warning(f"Expected 24 hours but got {len(pred_features_df)} hours for {target_day}")
            if len(pred_features_df) == 0:
                return {
                    'error': True,
                    'message': f'No data available for prediction date {target_day}',
                    'prediction_date': target_day
                }

        # 5. Get features required by the model
        excluded_cols = ['date', 'rain', 'rain_category']
        
        # Use scaler's feature names if available
        if hasattr(SCALER, 'feature_names_in_'):
            scaler_features = list(SCALER.feature_names_in_)
            logger.info(f"Using {len(scaler_features)} features required by the model")

            # Check for missing features
            feature_cols = [col for col in pred_features_df.columns if col not in excluded_cols]
            missing_features = [feat for feat in scaler_features if feat not in feature_cols]
            if missing_features:
                logger.error(f"Missing features required by the scaler: {missing_features}")
                return {
                    'error': True,
                    'message': f'Missing features required by model: {", ".join(missing_features[:5])}{"..." if len(missing_features) > 5 else ""}',
                    'prediction_date': target_day
                }

            X = pred_features_df[scaler_features].copy()
        else:
            # Fallback if scaler doesn't have feature names
            logger.warning("Scaler does not have feature_names_in_ attribute. Using all available features.")
            feature_cols = [col for col in pred_features_df.columns if col not in excluded_cols]
            X = pred_features_df[feature_cols].copy()

        # 6. Handle missing values
        if X.isna().any().any():
            logger.warning(f"Found NaN values in features, attempting to impute")

            # For lag features, try to use the original features
            for col in X.columns:
                if '_lag_' in col and X[col].isna().any():
                    base_col = col.split('_lag_')[0]
                    if base_col in X.columns:
                        X[col] = X[col].fillna(X[base_col])

            # For remaining NaNs, use forward/backward fill or zeros
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # 7. Scale and predict
        logger.info(f"Scaling and predicting for {len(X)} samples")
        X_scaled = SCALER.transform(X)
        predictions = MODEL.predict(X_scaled)
        probas = MODEL.predict_proba(X_scaled)

        # 8. Combine results with original data
        results_df = pred_features_df[['date', 'hour']].copy()
        if 'rain' in pred_features_df.columns:
            results_df['rain'] = pred_features_df['rain']
        results_df['rain_category'] = predictions
        results_df['prediction_confidence'] = np.max(probas, axis=1)

        # Map prediction to category label
        rain_categories = {
            0: "No rain",
            1: "Weak rain",
            2: "Moderate rain",
            3: "Heavy rain",
            4: "Severe rain"
        }

        results_df['rain_category_label'] = results_df['rain_category'].map(rain_categories)

        # Sort results by hour for better readability
        results_df = results_df.sort_values('date')

        # Create a summary for display
        summary = results_df.groupby('rain_category_label').size().reset_index()
        summary.columns = ['rain_category', 'hours_count']
        summary['percentage'] = (summary['hours_count'] / summary['hours_count'].sum() * 100).round(1)
        
        # Convert summary to dict for easy access in templates
        summary_dict = summary.set_index('rain_category').to_dict('index')
        
        # Convert DataFrame to list of dictionaries for JSON serialization
        hourly_predictions = []
        for _, row in results_df.iterrows():
            prediction = {
                'date': row['date'].isoformat(),
                'hour': int(row['hour']),
                'rain_category': int(row['rain_category']),
                'rain_category_label': row['rain_category_label'],
                'prediction_confidence': float(row['prediction_confidence'])
            }
            if 'rain' in results_df.columns:
                prediction['actual_rain'] = float(row['rain'])
            hourly_predictions.append(prediction)
        
        return {
            'error': False,
            'hourly_predictions': hourly_predictions,
            'summary': summary_dict,
            'prediction_date': target_day,
            'grid_id': nearest_grid_id if nearest_grid_id else None,
            'coordinates': (latitude, longitude)
        }

    except Exception as e:
        logger.error(f"Rainfall prediction failed: {str(e)}")
        logger.exception("Detailed traceback:")
        return {
            'error': True,
            'message': str(e),
            'prediction_date': pd.Timestamp.now().date() if prediction_date is None else pd.to_datetime(prediction_date).date()
        }