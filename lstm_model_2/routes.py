from flask import Blueprint, request, jsonify, send_file
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from datetime import datetime, timedelta
from model1 import predict as predict_price
from model2 import predict as predict_demand
from model3 import predict as predict_production

api_blueprint = Blueprint('api', __name__)

OPENWEATHER_API_KEY = 'f8eecee3f82a3ee3e9f0e4123b8c7bd7'
STATIC_FOLDER = 'static/'

# Ensure the static folder exists
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)


@api_blueprint.route('/predict', methods=['POST'])
def predict():
    data = request.json
    location = data['location']
    timestamp = data['timestamp']

    weather_data = fetch_weather_data(location, timestamp)
    features = extract_features(weather_data)

    demand = predict_demand(features)
    production = predict_production(features)
    price = predict_price(features)

    # Apply peak demand price surge if demand is highest in the day
    peak_hour = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ').hour
    if peak_hour >= 18 and peak_hour <= 21:
        price *= 1.2

    # Calculate surplus energy
    surplus = production - demand
    calculated_price = demand / 300

    # Generate and save hourly forecast graph
    hourly_graph = generate_hourly_graph(location, weather_data)

    # Generate and save 7-day forecast graph
    daily_graph = generate_daily_graph(location, weather_data)

    return jsonify({
        'energy_demand': demand,
        'energy_produced': production,
        'surplus': surplus,
        'price': calculated_price,
        'hourly_graph': f'/{hourly_graph}',
        'daily_graph': f'/{daily_graph}'
    })


def fetch_weather_data(location, timestamp):
    url = f'https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={OPENWEATHER_API_KEY}&units=metric'
    response = requests.get(url)
    data = response.json()
    return data


def extract_features(weather_data):
    # Extract necessary features from the weather API
    features = [
        weather_data['list'][0]['main']['temp'],
        weather_data['list'][0]['main']['feels_like'],
        weather_data['list'][0]['main']['humidity'],
        weather_data['list'][0]['main']['pressure'],
        weather_data['list'][0]['wind']['speed'],
        weather_data['list'][0]['wind']['deg'],
        weather_data['list'][0]['clouds']['all'],
        weather_data['list'][0]['rain']['3h'] if 'rain' in weather_data['list'][0] else 0
    ]
    return features


def generate_hourly_graph(location, weather_data):
    timestamps = []
    demand = []
    production = []
    price = []

    # Predict for the next 24 hours
    for i in range(24):
        hour_data = weather_data['list'][i]
        features = extract_features({'list': [hour_data]})
        demand.append(predict_demand(features))
        production.append(predict_production(features))
        price.append(predict_price(features))

        # Adjust peak hour price surge
        hour = datetime.strptime(hour_data['dt_txt'], '%Y-%m-%d %H:%M:%S').hour
        if hour >= 18 and hour <= 21:
            price[-1] *= 1.2

        timestamps.append(hour_data['dt_txt'])

    # Plot graph using seaborn
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=timestamps, y=demand, label='Energy Demand (kWh)', color='red')
    sns.lineplot(x=timestamps, y=production, label='Energy Production (kWh)', color='green')
    sns.lineplot(x=timestamps, y=price, label='Price (USD/kWh)', color='blue')

    plt.xticks(rotation=45)
    plt.title(f'Hourly Energy Forecast - {location}')
    plt.xlabel('Time')
    plt.ylabel('Energy/Price')
    plt.legend()

    # Save graph
    filename = f'{STATIC_FOLDER}hourly_forecast_{location}.png'
    plt.savefig(filename)
    plt.close()

    return filename


def generate_daily_graph(location, weather_data):
    dates = []
    daily_demand = []
    daily_price = []

    # Predict for the next 7 days
    for i in range(0, 40, 8):  # Every 8 timestamps = 1 day
        day_data = weather_data['list'][i]
        features = extract_features({'list': [day_data]})
        demand = predict_demand(features)
        price = predict_price(features)

        # Apply peak demand surge if necessary
        price *= 1.2 if demand > 2000 else 1

        dates.append(day_data['dt_txt'].split()[0])
        daily_demand.append(demand)
        daily_price.append(price)

    # Plot graph
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=dates, y=daily_demand, label='Daily Energy Demand (kWh)', color='red')
    sns.lineplot(x=dates, y=daily_price, label='Daily Price (USD/kWh)', color='blue')

    plt.xticks(rotation=45)
    plt.title(f'7-Day Energy Forecast - {location}')
    plt.xlabel('Date')
    plt.ylabel('Energy/Price')
    plt.legend()

    # Save graph
    filename = f'{STATIC_FOLDER}daily_forecast_{location}.png'
    plt.savefig(filename)
    plt.close()

    return filename
