# routes.py
from flask import Blueprint, request, jsonify, render_template
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from model2 import production_model, FEATURES_PROD, predict_energy_produced
from model import model, FEATURES, predict_energy_demand
import matplotlib
matplotlib.use('Agg')


# Replace with your actual OpenWeather API key
OPENWEATHER_API_KEY = "f8eecee3f82a3ee3e9f0e4123b8c7bd7"

bp = Blueprint('main', __name__)

# Set your base selling price per MW (you can change it later via API)
BASE_PRICE_PER_MW = 8  # ₹8 per MW


@bp.route('/')
def index():
    return render_template('index.html')


def get_coordinates(location):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    response = requests.get(url, params=params)
    data = response.json()
    return data["coord"]["lat"], data["coord"]["lon"]


def get_daily_forecast(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": lat, "lon": lon, "units": "metric", "appid": OPENWEATHER_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    forecasts = data.get("list", [])
    daily_forecasts = [f for f in forecasts if "18:00:00" in f.get("dt_txt", "")]
    return daily_forecasts[:5]


def month_to_season(month):
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[month]


def prepare_forecast_features(forecasts):
    rows = []
    for forecast in forecasts:
        dt = datetime.fromtimestamp(forecast["dt"])
        rows.append({
            "timestamp": dt,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "day_of_week": dt.weekday(),
            "season": month_to_season(dt.month),
            "temperature": forecast["main"]["temp"],
            "humidity": forecast["main"]["humidity"],
            "pressure": forecast["main"]["pressure"],
            "wind_speed": forecast["wind"]["speed"],
            "cloud_coverage": forecast["clouds"]["all"],
            "precipitation": forecast.get("rain", {}).get("3h", 0)
        })
    return pd.DataFrame(rows)


@bp.route('/predict_revenue', methods=["POST"])
def predict_revenue():
    """
    Predict revenue from energy production based on:
    - Energy Produced (MW)
    - Dynamic Pricing (₹/MW)
    """
    try:
        data = request.get_json()
        location = data.get("location")
        price_per_mw = data.get("price_per_mw", BASE_PRICE_PER_MW)

        # Fetch weather data and make predictions
        lat, lon = get_coordinates(location)
        daily_forecasts = get_daily_forecast(lat, lon)
        df_features = prepare_forecast_features(daily_forecasts)

        # Predict production
        X_future_production = df_features[FEATURES_PROD]
        production_predictions = predict_energy_produced(X_future_production)
        df_features["predicted_energy_production"] = production_predictions

        # Calculate revenue
        revenue = np.sum(production_predictions) * price_per_mw
        df_features["revenue"] = production_predictions * price_per_mw

        return jsonify({
            "total_revenue": round(revenue, 2),
            "price_per_mw": price_per_mw,
            "predicted_production": list(production_predictions)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/predict_wastage', methods=["POST"])
def predict_wastage():
    """
    Predict energy wastage if production > demand
    """
    try:
        data = request.get_json()
        location = data.get("location")
        price_per_mw = data.get("price_per_mw", BASE_PRICE_PER_MW)

        # Fetch weather data and make predictions
        lat, lon = get_coordinates(location)
        daily_forecasts = get_daily_forecast(lat, lon)
        df_features = prepare_forecast_features(daily_forecasts)

        # Predict demand and production
        X_future_demand = df_features[FEATURES]
        X_future_production = df_features[FEATURES_PROD]
        demand_predictions = predict_energy_demand(X_future_demand)
        production_predictions = predict_energy_produced(X_future_production)

        # Calculate surplus/deficit
        df_features["predicted_energy_demand"] = demand_predictions
        df_features["predicted_energy_production"] = production_predictions
        df_features["energy_surplus"] = production_predictions - demand_predictions

        # Calculate energy wastage
        energy_wastage = np.sum(production_predictions - demand_predictions)
        if energy_wastage < 0:
            energy_wastage = 0

        # Suggest lower pricing if wastage > 0
        suggestion = "No wastage" if energy_wastage == 0 else f"Sell at ₹{price_per_mw - 2}/MW to clear inventory."

        return jsonify({
            "total_energy_produced": np.sum(production_predictions),
            "total_energy_demand": np.sum(demand_predictions),
            "total_wastage": round(energy_wastage, 2),
            "suggestion": suggestion
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/update_price', methods=["POST"])
def update_price():
    """
    Allows the seller to update price per MW
    """
    global BASE_PRICE_PER_MW
    data = request.get_json()
    new_price = data.get("new_price")
    if not new_price:
        return jsonify({"error": "Provide 'new_price'"}), 400

    BASE_PRICE_PER_MW = new_price
    return jsonify({"message": f"Price per MW updated to ₹{new_price}"})


@bp.route('/predict_combined', methods=["POST"])
def predict_combined():
    """
    Predict revenue and wastage in a single request
    """
    data = request.get_json()
    location = data.get("location")
    price_per_mw = data.get("price_per_mw", BASE_PRICE_PER_MW)

    lat, lon = get_coordinates(location)
    daily_forecasts = get_daily_forecast(lat, lon)
    df_features = prepare_forecast_features(daily_forecasts)

    # Predictions
    demand_predictions = predict_energy_demand(df_features[FEATURES])
    production_predictions = predict_energy_produced(df_features[FEATURES_PROD])

    # Revenue
    total_revenue = np.sum(production_predictions) * price_per_mw
    energy_wastage = np.sum(production_predictions - demand_predictions)

    return jsonify({
        "total_revenue": round(total_revenue, 2),
        "total_energy_produced": np.sum(production_predictions),
        "total_energy_demand": np.sum(demand_predictions),
        "total_wastage": round(energy_wastage, 2),
        "suggested_price": price_per_mw - 2 if energy_wastage > 0 else price_per_mw
    })
