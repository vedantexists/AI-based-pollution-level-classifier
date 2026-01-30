from flask import Flask, request, jsonify
import requests
import joblib
import numpy as np

# Create Flask App
app = Flask(__name__)

# Load ML Model
model = joblib.load("model.pkl")

# OpenWeather API Key
API_KEY = "e5b23f2b70dd03f85bce1496c099698e"

https://github.com/vedantexists/AI-based-pollution-level-classifier.git
# -----------------------------
# City → Latitude & Longitude
# -----------------------------
def get_coordinates(city):

    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"

    res = requests.get(url)
    data = res.json()

    if len(data) == 0:
        return None, None

    lat = data[0]["lat"]
    lon = data[0]["lon"]

    return lat, lon


# -----------------------------
# Coordinates → Pollution Data
# -----------------------------
def get_pollution(lat, lon):

    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

    res = requests.get(url)
    data = res.json()

    return data["list"][0]["components"]

@app.route("/")
def home():
    return "Air Quality API is Running!"

# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        # Option 1: Manual Values
        if "pm2_5" in data:

            pm2_5 = data["pm2_5"]
            pm10 = data["pm10"]
            no2 = data["no2"]
            so2 = data["so2"]
            co = data["co"]
            o3 = data["o3"]

        # Option 2: City Name
        elif "city" in data:

            city = data["city"]

            lat, lon = get_coordinates(city)

            if lat is None:
                return jsonify({"error": "City not found"}), 404

            pollution = get_pollution(lat, lon)

            pm2_5 = pollution["pm2_5"]
            pm10 = pollution["pm10"]
            no2 = pollution["no2"]
            so2 = pollution["so2"]
            co = pollution["co"]
            o3 = pollution["o3"]

        else:
            return jsonify({"error": "Invalid input"}), 400


        # Prepare input for model
        X = np.array([[pm2_5, pm10, no2, so2, co, o3]])

        # Predict
        prediction = model.predict(X)[0]

        return jsonify({
            "prediction": str(prediction),
            "inputs": {
                "pm2_5": pm2_5,
                "pm10": pm10,
                "no2": no2,
                "so2": so2,
                "co": co,
                "o3": o3
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Start Server
# -----------------------------
if __name__ == "__main__":

    app.run(debug=True)
