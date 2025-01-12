import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

# Data Extraction
# Assuming weather_data.csv, solar_output.csv, and metadata.csv are the input datasets
weather_data = pd.read_csv("weather_data.csv")
solar_output = pd.read_csv("solar_output.csv")
metadata = pd.read_csv("metadata.csv")

# Merge the datasets
solar_output = pd.merge(solar_output, metadata, on='PV Serial Number', how='inner')

# Data Preprocessing
# Convert dates to datetime format
solar_output['Date'] = pd.to_datetime(solar_output['Date'])
weather_data['Date'] = pd.to_datetime(weather_data['Date'])

# Merge weather data with solar output
merged_data = pd.merge(solar_output, weather_data, on=['Date', 'Latitude', 'Longitude'], how='inner')

# Handle missing values
merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')

# Feature Engineering
# Normalize energy output by installed power
merged_data['Specific Energy (kWh/kWp)'] = merged_data['Produced Energy (kWh)'] / merged_data['Installed Power (kWp)']

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
sns.heatmap(merged_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Plot target variable distribution
plt.figure(figsize=(10, 5))
sns.histplot(merged_data['Specific Energy (kWh/kWp)'], kde=True, bins=30)
plt.title("Specific Energy Distribution")
plt.show()

# Time series analysis
plt.figure(figsize=(15, 5))
plt.plot(merged_data['Date'], merged_data['Specific Energy (kWh/kWp)'], label='Specific Energy')
plt.title("Specific Energy Over Time")
plt.xlabel("Date")
plt.ylabel("Specific Energy (kWh/kWp)")
plt.legend()
plt.show()

# Feature Selection
X = merged_data[['Shortwave Radiation (W/m^2)', 'Direct Radiation (W/m^2)', 'Diffuse Radiation (W/m^2)',
                 'Temperature (\u00b0C)', 'Relative Humidity (%)', 'Cloud Cover (%)',
                 'Rain (mm)', 'Wind Speed (m/s)', 'Latitude', 'Longitude']]
y = merged_data['Specific Energy (kWh/kWp)']

# Select top features
selector = SelectKBest(score_func=f_regression, k=8)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("Selected Features:", selected_features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Model Building
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Model Evaluation
# Linear Regression Metrics
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_r2 = r2_score(y_test, lr_preds)
print("Linear Regression RMSE:", lr_rmse)
print("Linear Regression R2:", lr_r2)

# Random Forest Metrics
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2 = r2_score(y_test, rf_preds)
print("Random Forest RMSE:", rf_rmse)
print("Random Forest R2:", rf_r2)

# Model Comparison
models = ['Linear Regression', 'Random Forest']
rmse_scores = [lr_rmse, rf_rmse]
r2_scores = [lr_r2, rf_r2]

plt.figure(figsize=(10, 5))
plt.bar(models, rmse_scores, color=['blue', 'green'])
plt.title("RMSE Comparison")
plt.ylabel("RMSE")
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(models, r2_scores, color=['blue', 'green'])
plt.title("R2 Score Comparison")
plt.ylabel("R2 Score")
plt.show()

# Save the better model
if rf_r2 > lr_r2:
    joblib.dump(rf_model, 'best_model.pkl')
    print("Random Forest model saved as best_model.pkl")
else:
    joblib.dump(lr_model, 'best_model.pkl')
    print("Linear Regression model saved as best_model.pkl")

# Real-time Prediction Function
def predict_energy(weather_data, metadata):
    """
    Predict energy generation for new location-specific weather data.

    Parameters:
    weather_data: pd.DataFrame - Real-time weather data (features: selected_features).
    metadata: dict - Metadata for the location (keys: Latitude, Longitude, Installed Power).

    Returns:
    Predicted Specific Energy (kWh/kWp) and total energy (kWh).
    """
    model = joblib.load('best_model.pkl')
    weather_data['Latitude'] = metadata['Latitude']
    weather_data['Longitude'] = metadata['Longitude']
    X_real_time = weather_data[selected_features]
    specific_energy = model.predict(X_real_time)
    total_energy = specific_energy * metadata['Installed Power']
    return specific_energy, total_energy

# Example usage:
# real_time_weather = pd.DataFrame({
#     'Shortwave Radiation (W/m^2)': [600],
#     'Direct Radiation (W/m^2)': [300],
#     'Diffuse Radiation (W/m^2)': [150],
#     'Temperature (\u00b0C)': [25],
#     'Relative Humidity (%)': [40],
#     'Cloud Cover (%)': [20],
#     'Rain (mm)': [0],
#     'Wind Speed (m/s)': [3]
# })
# metadata = {'Latitude': 28.7041, 'Longitude': 77.1025, 'Installed Power': 5.0}
# specific_energy, total_energy = predict_energy(real_time_weather, metadata)
# print(f"Specific Energy: {specific_energy} kWh/kWp")
# print(f"Total Energy: {total_energy} kWh")
