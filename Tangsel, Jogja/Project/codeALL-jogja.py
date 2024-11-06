import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, GRU, Dense, SimpleRNN
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

# Prepare the data for time series models with sequence length of 5 days
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

# Function to evaluate the model for each pollutant with additional metrics
def evaluate_model_per_pollutant(y_true, y_pred):
    metrics = {}
    for i, pollutant in enumerate(numeric_columns):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])) * 100
        mbd = np.mean(y_pred[:, i] - y_true[:, i])
        smape = 100/len(y_true) * np.sum(2 * np.abs(y_pred[:, i] - y_true[:, i]) / (np.abs(y_true[:, i]) + np.abs(y_pred[:, i])))
        metrics[pollutant] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R-squared': r2, 'MAPE': mape, 'MBD': mbd, 'sMAPE': smape}
    return metrics

# CNN Model
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(seq_length, X.shape[2])))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(y.shape[1]))
cnn_model.compile(optimizer='adam', loss='mean_squared_error')
cnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
cnn_predictions_scaled = cnn_model.predict(X_test)
cnn_predictions = scaler_target.inverse_transform(cnn_predictions_scaled)
cnn_metrics = evaluate_model_per_pollutant(y_test, cnn_predictions_scaled)

# RNN Model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(256, return_sequences=True, input_shape=(seq_length, X.shape[2])))
rnn_model.add(SimpleRNN(64))
rnn_model.add(Dense(y.shape[1]))
rnn_model.compile(optimizer='adam', loss='mean_squared_error')
rnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
rnn_predictions_scaled = rnn_model.predict(X_test)
rnn_predictions = scaler_target.inverse_transform(rnn_predictions_scaled)
rnn_metrics = evaluate_model_per_pollutant(y_test, rnn_predictions_scaled)

# GRU Model
gru_model = Sequential()
gru_model.add(GRU(256, return_sequences=True, input_shape=(seq_length, X.shape[2])))
gru_model.add(GRU(64))
gru_model.add(Dense(y.shape[1]))
gru_model.compile(optimizer='adam', loss='mean_squared_error')
gru_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
gru_predictions_scaled = gru_model.predict(X_test)
gru_predictions = scaler_target.inverse_transform(gru_predictions_scaled)
gru_metrics = evaluate_model_per_pollutant(y_test, gru_predictions_scaled)

# LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(256, return_sequences=True, input_shape=(seq_length, X.shape[2])))
lstm_model.add(LSTM(64))
lstm_model.add(Dense(y.shape[1]))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
lstm_predictions_scaled = lstm_model.predict(X_test)
lstm_predictions = scaler_target.inverse_transform(lstm_predictions_scaled)
lstm_metrics = evaluate_model_per_pollutant(y_test, lstm_predictions_scaled)

# SARIMA Model (only for PM2.5)
sarima_model = SARIMAX(data['PM2.5'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)
sarima_predictions = sarima_fit.predict(start=len(data) - len(y_test), end=len(data) - 1)
sarima_metrics = {
    'PM2.5': {
        'MSE': mean_squared_error(y_test[:, 0], sarima_predictions),
        'MAE': mean_absolute_error(y_test[:, 0], sarima_predictions),
        'RMSE': np.sqrt(mean_squared_error(y_test[:, 0], sarima_predictions)),
        'R-squared': r2_score(y_test[:, 0], sarima_predictions),
        'MAPE': np.mean(np.abs((y_test[:, 0] - sarima_predictions) / y_test[:, 0])) * 100,
        'MBD': np.mean(sarima_predictions - y_test[:, 0]),
        'sMAPE': 100/len(y_test) * np.sum(2 * np.abs(sarima_predictions - y_test[:, 0]) / (np.abs(y_test[:, 0]) + np.abs(sarima_predictions)))
    }
}

# Print evaluation metrics for each pollutant and model
print("CNN Metrics:")
for pollutant in cnn_metrics:
    print(f"{pollutant} - MSE: {cnn_metrics[pollutant]['MSE']}, MAE: {cnn_metrics[pollutant]['MAE']}, RMSE: {cnn_metrics[pollutant]['RMSE']}, R-squared: {cnn_metrics[pollutant]['R-squared']}, MAPE: {cnn_metrics[pollutant]['MAPE']}, MBD: {cnn_metrics[pollutant]['MBD']}, sMAPE: {cnn_metrics[pollutant]['sMAPE']}")

print("\nRNN Metrics:")
for pollutant in rnn_metrics:
    print(f"{pollutant} - MSE: {rnn_metrics[pollutant]['MSE']}, MAE: {rnn_metrics[pollutant]['MAE']}, RMSE: {rnn_metrics[pollutant]['RMSE']}, R-squared: {rnn_metrics[pollutant]['R-squared']}, MAPE: {rnn_metrics[pollutant]['MAPE']}, MBD: {rnn_metrics[pollutant]['MBD']}, sMAPE: {rnn_metrics[pollutant]['sMAPE']}")

print("\nGRU Metrics:")
for pollutant in gru_metrics:
    print(f"{pollutant} - MSE: {gru_metrics[pollutant]['MSE']}, MAE: {gru_metrics[pollutant]['MAE']}, RMSE: {gru_metrics[pollutant]['RMSE']}, R-squared: {gru_metrics[pollutant]['R-squared']}, MAPE: {gru_metrics[pollutant]['MAPE']}, MBD: {gru_metrics[pollutant]['MBD']}, sMAPE: {gru_metrics[pollutant]['sMAPE']}")

print("\nLSTM Metrics:")
for pollutant in lstm_metrics:
    print(f"{pollutant} - MSE: {lstm_metrics[pollutant]['MSE']}, MAE: {lstm_metrics[pollutant]['MAE']}, RMSE: {lstm_metrics[pollutant]['RMSE']}, R-squared: {lstm_metrics[pollutant]['R-squared']}, MAPE: {lstm_metrics[pollutant]['MAPE']}, MBD: {lstm_metrics[pollutant]['MBD']}, sMAPE: {lstm_metrics[pollutant]['sMAPE']}")

print("\nSARIMA Metrics (only for PM2.5):")
for pollutant in sarima_metrics:
    print(f"{pollutant} - MSE: {sarima_metrics[pollutant]['MSE']}, MAE: {sarima_metrics[pollutant]['MAE']}, RMSE: {sarima_metrics[pollutant]['RMSE']}, R-squared: {sarima_metrics[pollutant]['R-squared']}, MAPE: {sarima_metrics[pollutant]['MAPE']}, MBD: {sarima_metrics[pollutant]['MBD']}, sMAPE: {sarima_metrics[pollutant]['sMAPE']}")

# Plot each pollutant over time and compare actual vs predicted values for test set
pollutants = ['PM2.5', 'PM10', 'SO2', 'CO', 'O3', 'NO2']
for pollutant in pollutants:
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(X_test):], scaler_target.inverse_transform(y_test)[:, pollutants.index(pollutant)], label=f'Actual {pollutant}')
    plt.plot(data.index[-len(X_test):], cnn_predictions[:, pollutants.index(pollutant)], label=f'CNN Predicted {pollutant}')
    plt.plot(data.index[-len(X_test):], rnn_predictions[:, pollutants.index(pollutant)], label=f'RNN Predicted {pollutant}')
    plt.plot(data.index[-len(X_test):], gru_predictions[:, pollutants.index(pollutant)], label=f'GRU Predicted {pollutant}')
    plt.plot(data.index[-len(X_test):], lstm_predictions[:, pollutants.index(pollutant)], label=f'LSTM Predicted {pollutant}')
    if pollutant == 'PM2.5':
        plt.plot(data.index[-len(X_test):], sarima_predictions, label=f'SARIMA Predicted {pollutant}')
    plt.title(f'{pollutant} over Time (Actual vs Predicted)')
    plt.xlabel('Date')
    plt.ylabel(pollutant)
    plt.legend()
    plt.show()
