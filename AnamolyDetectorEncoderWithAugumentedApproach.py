import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Load the datasets
input_file = 'SourceDataSet_AllRecord_FromSource.csv'
subset_file = 'MissingRecord_MissingColumnData.csv'

input_df = pd.read_csv(input_file)
subset_df = pd.read_csv(subset_file)

# Ensure both dataframes have the same columns
assert list(input_df.columns) == list(subset_df.columns), "Input and subset dataframes must have the same columns"

# Augment the data with missing indicators
def augment_data(df):
    missing_indicator = df.isna().astype(int)
    df_filled = df.fillna(0)
    augmented_df = pd.concat([df_filled, missing_indicator], axis=1)
    return augmented_df

# Preprocess the data: Fill missing values and standardize
def preprocess_data(df, scaler=None):
    augmented_df = augment_data(df)
    augmented_df.to_csv('agumentedData.csv', index=False)
    if scaler:
        scaled_df = scaler.transform(augmented_df)
    else:
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(augmented_df)
    return scaled_df, scaler

# Preprocess input data
input_data, input_scaler = preprocess_data(input_df)

# Define the autoencoder model
input_dim = input_data.shape[1]
encoding_dim = 14  # Can be adjusted

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(input_data, input_data, epochs=1000, batch_size=50, shuffle=True, validation_split=0.2)

# Preprocess subset data using the same scaler
subset_data, _ = preprocess_data(subset_df, scaler=input_scaler)

# Predict using the autoencoder
reconstructed_data = autoencoder.predict(subset_data)

# Calculate the reconstruction error
reconstruction_error = np.mean(np.power(subset_data - reconstructed_data, 2), axis=1)

# Set a threshold for anomaly detection
threshold = np.percentile(reconstruction_error, 100)

# Identify anomalies based on reconstruction error
anomalies = reconstruction_error > threshold

# Detect rows with missing values
missing_values_anomalies = subset_df.isna().any(axis=1)

# Combine anomalies with missing value indicators
anomalies_combined = anomalies | missing_values_anomalies

# Extract the anomalous data
anomalous_data = subset_df[anomalies_combined]

print(f"Anomalies detected:\n{anomalous_data}")

# Save anomalies to a file if needed
anomalous_data.to_csv('anomalies.csv', index=False)
