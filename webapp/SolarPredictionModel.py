#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import joblib
from pymongo import MongoClient, DeleteMany
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
print('inside')
warnings.filterwarnings('ignore')
client = MongoClient("mongodb://localhost:27017/")
db = client["solar_energy_db"]
metadata_collection = db["metadata"]
solarproduction_collection = db["solarproduction"]
weather_collection = db["weather"]

# Real-time Prediction Function
def predict_energy(weather_data, metadata):
    model = joblib.load('best_model.pkl')
    scaler = joblib.load("scaler.pkl")
    selected_features = joblib.load("features.pkl")
    weather_data['Latitude'] = metadata['latitude']
    weather_data['Longitude'] = metadata['longitude']
    X_real_time = scaler.transform(weather_data[selected_features])
    X_real_time = pd.DataFrame(X_real_time, columns=selected_features)
    #X_real_time = pd.DataFrame(weather_data[selected_features], columns=selected_features)
    specific_energy = model.predict(X_real_time)
    total_energy = specific_energy * float(metadata['installed_power'])
    return specific_energy, total_energy

def data_processing(df):
    df.rename(columns={'temperature_2m (°C)':'temperature','relative_humidity_2m (%)':'relative_humidity',
                    'rain (mm)':'rain','cloud_cover (%)':'cloud_cover','wind_speed_10m (km/h)':'wind_speed',
                    'shortwave_radiation (W/m²)':'shortwave_radiation','direct_radiation (W/m²)':'direct_radiation',
                    'diffuse_radiation (W/m²)':'diffuse_radiation'},inplace=True)
    df.reset_index(drop = True,inplace=True)
    df.rename(columns={"time": "Date"},inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    data_dict = df.to_dict("records")
    return df, data_dict

def data_transform(df):
    df['Hour'] = df['Date'].dt.hour
    df['Month'] = df['Date'].dt.month
    df['Date_only'] = df['Date'].dt.date.astype(str)
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    for lag in [1, 2, 3]:
        df[f'Lag_{lag}_Radiation'] = df['shortwave_radiation'].shift(lag)
    df['Rolling_Temp_Mean'] = df['temperature'].rolling(window=3).mean()
    df['Radiation_CloudCover'] = df['shortwave_radiation'] * df['cloud_cover']
    result = (
        df.groupby('Date_only')
        .apply(lambda x: x.drop(columns=['Date', 'Date_only']).to_dict(orient='records'))
        .to_dict()
    )
    return result

def model_prediciton(result,metadata):
    total_list = []
    required_features = joblib.load("features.pkl").to_list()
    for date, records in result.items():
        total_energy_total = 0
        for record in records:
            filtered_data = []
            filtered_data.append({key: record[key] for key in required_features})
            # Convert to DataFrame
            real_time_weather = pd.DataFrame(filtered_data).fillna(0)
            #metadata = {'Latitude': 28.7041, 'Longitude': 77.1025, 'Installed Power': 310.0}
            specific_energy, total_energy = predict_energy(real_time_weather, metadata)
            total_energy_total = total_energy_total + total_energy
        total = {'Date':date,'Total_Solar_Energy':total_energy_total[0]}
        total_list.append(total)
    res_df = pd.DataFrame(total_list)
    res_df['Timestamp'] = pd.Timestamp.now()
    return res_df

def mongo_upload(data_dict,collection):
    result = collection.insert_many(data_dict)
    print(f'Inserted acknowledgement is {result.acknowledged}')
    
def mongo_fetch(plant_name):
    metadata = metadata_collection.find_one({"plant_name":plant_name})
    return metadata

def mongo_delete_many(delete_many,collection):
    result = collection.bulk_write(
        delete_many
    )
    print(result)





