
# Crop Yield Prediction using Random Forest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("crop_yield_data.csv")

# Encode categorical data
df['Crop'] = df['Crop'].astype('category').cat.codes

# Features and target
X = df[['Crop', 'Area', 'Rainfall', 'Temperature']]
y = df['Yield']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prediction and evaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Predicting a new sample
sample = [[0, 3.0, 180, 31]]  # Rice, Area=3.0, Rainfall=180mm, Temp=31°C
predicted_yield = model.predict(sample)
print(f"Predicted Crop Yield: {predicted_yield[0]:.2f} tons")
