# ============================================
# Household Energy Optimization System
# AI/ML Capstone Project
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------
# 1. Load Dataset
# ----------------------------
print("Loading Dataset...")

df = pd.read_csv(
    "household_power_consumption.txt",
    sep=';',
    nrows=50000,
    low_memory=False
)

print("\nFirst 5 Rows:")
print(df.head())

# ----------------------------
# 2. Data Preprocessing (FIXED)
# ----------------------------

print("\nCleaning Data...")

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert all columns except Date & Time to numeric
for col in df.columns:
    if col not in ['Date', 'Time']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop missing values
df.dropna(inplace=True)

# Convert Date properly
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# Create Hour feature
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour

print("\nData Types After Cleaning:")
print(df.dtypes)

# ----------------------------
# 3. Feature Selection
# ----------------------------

features = [
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3',
    'Hour'
]

X = df[features]
y = df['Global_active_power']

# ----------------------------
# 4. Train-Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 5. Model Training
# ----------------------------

print("\nTraining Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ----------------------------
# 6. Prediction
# ----------------------------

lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# ----------------------------
# 7. Evaluation
# ----------------------------

print("\nModel Evaluation Results:")

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("\nLinear Regression:")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)

print("\nRandom Forest:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)

# ----------------------------
# 8. Visualization
# ----------------------------

y_test_reset = y_test.reset_index(drop=True)

plt.figure(figsize=(10,5))
plt.plot(y_test_reset[:100], label="Actual")
plt.plot(rf_pred[:100], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Energy Consumption")

plt.savefig("prediction_output.png")
print("\nGraph saved as prediction_output.png")

plt.close()

print("\nProject Completed Successfully!")
