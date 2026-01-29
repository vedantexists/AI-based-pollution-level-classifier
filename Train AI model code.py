import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the city_day file
df = pd.read_csv('city_day.csv')

# 2. Select modern pollutant features
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
X = df[features]
y = df['AQI_Bucket'] # Your target category

# 3. Handling Missing Values
X = X.fillna(X.mean())
y = y.fillna('Moderate') # Filling missing labels with the most common class

# 4. Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Save the 'Brain'
joblib.dump(model, 'pollution_model.pkl')
print("Phase 1 Success: city_day model is ready!")

# ... (after model.fit)
import joblib

# Save the model
joblib.dump(model, 'pollution_model.pkl')

# Save the scaler (Crucial for the website!)
joblib.dump(scaler, 'scaler.pkl') 

print("Both model and scaler saved successfully!")

