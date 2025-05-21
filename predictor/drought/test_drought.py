
from drought_pipeline import run_drought_forecast

if __name__ == "__main__":
    # Sample coordinates to test each region
    test_coords = {
        'Sindh': (25.3960, 68.3578),        # Hyderabad
        'Punjab': (31.5497, 74.3436),       # Lahore
        'Balochistan': (30.1798, 66.9750),  # Quetta
        'KPK': (34.0151, 71.5249),          # Peshawar
        'Outside (Fallback)': (35.8617, 76.5133)  # Gilgit (should fallback to Sindh model)
    }

    for region, (lat, lon) in test_coords.items():
        print(f"\n--- Testing for {region} ({lat}, {lon}) ---")
        try:
            results = run_drought_forecast(lat, lon)
            print(f"First Prediction: {results[0]['drought_category']} ({results[0]['confidence']*100:.2f}%)")
        except Exception as e:
            print(f"Error while processing {region}: {e}")
