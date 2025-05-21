from django.urls import path
from . import views
from .views import heatwave_prediction_view
from .views import drought_forecast_view
from .views import get_rainfall_prediction

urlpatterns = [
    path('test/', views.test_view, name='test'),
     path("heatwave/", heatwave_prediction_view, name="predict_heatwave"),
     path('drought-forecast/', drought_forecast_view, name='drought_forecast'),
     path('predict-rainfall/', get_rainfall_prediction, name='rainfall_prediction'),
    

]
