#model2.py
import xgboost as xgb

# Define the feature order expected by the production model.
# (Assuming it uses the same features as your demand model, adjust if necessary.)
FEATURES_PROD = [
    'hour', 
    'day_of_week', 
    'day', 
    'month', 
    'year', 
    'season',
    'temperature', 
    'feels_like', 
    'humidity', 
    'pressure',
    'wind_speed', 
    'wind_direction', 
    'cloud_coverage', 
    'precipitation'
]

# Load the pre-trained XGBoost production model from model2.pkl.
production_model = xgb.XGBRegressor()
production_model.load_model("model2.pkl")

def predict_energy_produced(X):
    """Predict energy produced for a given feature DataFrame X."""
    return production_model.predict(X)
