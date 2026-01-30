import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Training data with 6 features
data = {
    "pm25": [30, 45, 60, 80, 120, 200, 300],
    "pm10": [40, 60, 90, 120, 180, 250, 350],
    "no2":  [15, 25, 35, 50, 70, 90, 120],
    "so2":  [5, 10, 15, 20, 30, 40, 50],
    "co":   [0.4, 0.6, 0.8, 1.2, 1.8, 2.5, 3.0],
    "o3":   [10, 20, 30, 45, 60, 80, 100],
    "aqi":  [50, 80, 110, 150, 200, 300, 400]
}

df = pd.DataFrame(data)

# Input features (6 now)
X = df[["pm25", "pm10", "no2", "so2", "co", "o3"]]
y = df["aqi"]

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained with 6 features and saved")
