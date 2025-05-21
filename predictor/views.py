from django.shortcuts import render

# Create your views here.

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import logging
import pandas as pd
from .heatwave_predictor import predict_heatwave
from .drought.drought_pipeline import run_drought_forecast
from .rainfall.rainfall_pipeline import predict_rainfall_24h

def test_view(request):
    return JsonResponse({"message": "Backend is working!"})


def heatwave_prediction_view(request):
    try:
        lat = float(request.GET.get("lat"))
        lon = float(request.GET.get("lon"))
        
        result = predict_heatwave(lat, lon)
        
        # Format response to match the structure from drought_forecast_view
        return JsonResponse({
            "predictions": result,
            "grid_info": f"Using nearest grid at ({lat:.5f}, {lon:.5f})"
        })
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid or missing coordinates."}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def drought_forecast_view(request):
    try:
        lat = float(request.GET.get('lat'))
        lon = float(request.GET.get('lon'))
        
        results = run_drought_forecast(lat, lon)
        
        # Format response to match React expectations
        return JsonResponse({
            "predictions": results,
            "grid_info": f"Using nearest grid at ({lat:.5f}, {lon:.5f})"
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    


logger = logging.getLogger(__name__)

@csrf_exempt
@require_http_methods(["GET", "POST"])  # Only allow GET and POST
def get_rainfall_prediction(request):
    """API endpoint to get rainfall predictions"""
    try:
        # Initialize variables
        latitude = None
        longitude = None
        prediction_date = None

        # Handle request data
        if request.method == 'GET':
            latitude = request.GET.get('latitude')
            longitude = request.GET.get('longitude')
            prediction_date = request.GET.get('prediction_date')
        elif request.method == 'POST':
            if request.content_type == 'application/json':
                try:
                    data = json.loads(request.body)
                except json.JSONDecodeError:
                    return JsonResponse({'error': 'Invalid JSON data'}, status=400)
            else:
                data = request.POST
            
            latitude = data.get('latitude')
            longitude = data.get('longitude')
            prediction_date = data.get('prediction_date')

        # Validate required parameters
        if not latitude or not longitude:
            return JsonResponse(
                {'error': 'Missing required parameters: latitude and longitude'},
                status=400
            )

        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (TypeError, ValueError):
            return JsonResponse(
                {'error': 'Latitude and longitude must be numbers'},
                status=400
            )

        # Validate coordinate ranges
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return JsonResponse(
                {'error': 'Invalid coordinates. Latitude must be between -90 and 90, longitude between -180 and 180'},
                status=400
            )

        # Make prediction
        prediction_results = predict_rainfall_24h(
            latitude=latitude,
            longitude=longitude,
            prediction_date=prediction_date  # Can be None
        )

        # Check for error in prediction results
        if prediction_results.get('error', False):
            return JsonResponse(
                {
                    'status': 'error',
                    'message': prediction_results.get('message', 'Failed to generate prediction')
                },
                status=400
            )
        
        # No need to iterate through DataFrame rows since we're already returning a list of dicts
        # from our updated predict_rainfall_24h function

        # Prepare the response
        response_data = {
            'status': 'success',
            'prediction_date': prediction_results['prediction_date'].strftime('%Y-%m-%d'),
            'coordinates': {
                'latitude': prediction_results['coordinates'][0],
                'longitude': prediction_results['coordinates'][1],
            },
            'grid_id': prediction_results['grid_id'],
            'hourly_predictions': prediction_results['hourly_predictions'],  # Already formatted for JSON
            'summary': prediction_results['summary']
        }

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return JsonResponse(
            {
                'status': 'error',
                'message': 'Failed to generate prediction',
                'error_details': str(e)
            },
            status=500
        )