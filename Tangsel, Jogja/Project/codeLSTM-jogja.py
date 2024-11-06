import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the datasets
data1 = pd.read_csv('jogja2023.csv', delimiter=';')
data2 = pd.read_csv('jogja2022.csv', delimiter=';')
data3 = pd.read_csv('jogja2021.csv', delimiter=';')
data4 = pd.read_csv('jogja2020.csv')

# Preprocess the data (assuming both datasets have the same structure)
data1['Date'] = pd.to_datetime(data1['Waktu'], format='%d/%m/%Y', errors='coerce')
data2['Date'] = pd.to_datetime(data2['Waktu'], format='%d/%m/%Y', errors='coerce')
data3['Date'] = pd.to_datetime(data3['Waktu'], format='%d/%m/%Y', errors='coerce')
data4['Date'] = pd.to_datetime(data4['Date'], format='%m/%d/%Y', errors='coerce')

# Drop rows with invalid dates
data1.dropna(subset=['Waktu'], inplace=True)
data2.dropna(subset=['Waktu'], inplace=True)
data3.dropna(subset=['Waktu'], inplace=True)
data4.dropna(subset=['Date'], inplace=True)

# Set the 'Date' column as the index for all datasets
data1.set_index('Waktu', inplace=True)
data2.set_index('Waktu', inplace=True)
data3.set_index('Waktu', inplace=True)
data4.set_index('Date', inplace=True)

# Select relevant columns from data4 and rename them to match the other datasets
data4 = data4[['PM10', 'SO2', 'CO', 'O3', 'NO2']]
data4['PM2.5'] = np.nan  # Add PM2.5 column with NaN values to match other datasets

# Concatenate the datasets
data = pd.concat([data1, data2, data3, data4])

# Replace commas with periods and '-' with NaN, then convert to float
for col in ['PM2.5', 'PM10', 'SO2', 'CO', 'O3', 'NO2']:
    data[col] = data[col].astype(str).str.replace(',', '.').replace('-', np.nan).astype(float)

# Remove rows with 0 data
data = data[(data[['PM2.5', 'PM10', 'SO2', 'CO', 'O3', 'NO2']] != 0).all(axis=1)]

# Handle NaN values in the numeric columns only
numeric_columns = ['PM2.5', 'PM10', 'SO2', 'CO', 'O3', 'NO2']
data[numeric_columns] = data[numeric_columns].fillna(method='ffill').fillna(method='bfill')

# Select features and target
features = data[numeric_columns]
target = data[numeric_columns]

# Normalize the features and target
scaler_features = MinMaxScaler()
features_scaled = scaler_features.fit_transform(features)

scaler_target = MinMaxScaler()
target_scaled = scaler_target.fit_transform(target)

# Prepare the data for LSTM with sequence length of 5 days
def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(features_scaled, target_scaled, seq_length)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(seq_length, X.shape[2])))
model.add(LSTM(64))
model.add(Dense(y.shape[1]))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions for pollutants (features)
predictions_scaled = model.predict(X_test)
predicted_pollutants = scaler_target.inverse_transform(predictions_scaled)
actual_pollutants = scaler_target.inverse_transform(y_test)

predicted_df = pd.DataFrame(predicted_pollutants, columns=features.columns)
actual_df = pd.DataFrame(actual_pollutants, columns=features.columns)

print("Predicted Pollutants:")
print(predicted_df.head(10))

print("Actual Pollutants:")
print(actual_df.head(10))

# Plot each pollutant over time and compare actual vs predicted values for test set
pollutants = ['PM2.5', 'PM10', 'SO2', 'CO', 'O3', 'NO2']
for pollutant in pollutants:
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(X_test):], actual_df[pollutant], label=f'Actual {pollutant}')
    plt.plot(data.index[-len(X_test):], predicted_df[pollutant], label=f'Predicted {pollutant}')
    plt.title(f'{pollutant} over Time (Actual vs Predicted)')
    plt.xlabel('Date')
    plt.ylabel(pollutant)
    plt.legend()
    plt.show()

# Calculate and print evaluation metrics for each pollutant
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

def mean_bias_deviation(y_true, y_pred):
    return np.mean(y_pred - y_true)

metrics = ['MSE', 'MAE', 'RMSE', 'R-squared', 'MAPE', 'MBD', 'sMAPE']
results = {metric: {} for metric in metrics}

for pollutant in numeric_columns:
    y_true = actual_df[pollutant]
    y_pred = predicted_df[pollutant]
    
    results['MSE'][pollutant] = mean_squared_error(y_true, y_pred)
    results['MAE'][pollutant] = mean_absolute_error(y_true, y_pred)
    results['RMSE'][pollutant] = np.sqrt(mean_squared_error(y_true, y_pred))
    results['R-squared'][pollutant] = r2_score(y_true, y_pred)
    results['MAPE'][pollutant] = mean_absolute_percentage_error(y_true, y_pred)
    results['MBD'][pollutant] = mean_bias_deviation(y_true, y_pred)
    results['sMAPE'][pollutant] = symmetric_mean_absolute_percentage_error(y_true, y_pred)

for metric in metrics:
    print(f"\n{metric}:")
    for pollutant in numeric_columns:
        print(f"{pollutant}: {results[metric][pollutant]}")
