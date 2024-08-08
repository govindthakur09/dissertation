import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_curve, auc

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the datasets
input_file = 'SourceDataSet_AllRecord_FromSource.csv'
subset_file = 'MissingRecord_MissingColumnData.csv'

input_df = pd.read_csv(input_file)
subset_df = pd.read_csv(subset_file)

# Ensure both dataframes have the same columns
assert list(input_df.columns) == list(subset_df.columns), "Input and subset dataframes must have the same columns"


# Augment the data
def augment_data(df, missing_row_indicator=False):
    missing_indicator = df.isna().astype(int)
    df_filled = df.fillna(0)
    augmented_df = pd.concat([df_filled, missing_indicator], axis=1)
    if missing_row_indicator:
        row_missing = df.isna().all(axis=1).astype(int).to_frame(name='row_missing')
        augmented_df = pd.concat([augmented_df, row_missing], axis=1)
    return augmented_df


# Preprocess the data
def preprocess_data(df, scaler=None, missing_row_indicator=False):
    augmented_df = augment_data(df, missing_row_indicator)
    if scaler:
        scaled_df = scaler.transform(augmented_df)
    else:
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(augmented_df)
    return scaled_df, scaler


# Preprocess input data
input_data, input_scaler = preprocess_data(input_df, missing_row_indicator=True)

# Define the autoencoder model
input_dim = input_data.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(input_data, input_data, epochs=4, batch_size=50, shuffle=True, validation_split=0.2)

# Preprocess subset data
subset_data, _ = preprocess_data(subset_df, scaler=input_scaler, missing_row_indicator=True)

# Predict using the autoencoder
reconstructed_data = autoencoder.predict(subset_data)

# Calculate the reconstruction error
reconstruction_error = np.mean(np.power(subset_data - reconstructed_data, 2), axis=1)

# Plot the reconstruction error distribution
plt.hist(reconstruction_error, bins=50, alpha=0.75, color='blue')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Reconstruction Errors')
plt.show()

# Experiment with different thresholds
thresholds = np.linspace(0.25, 1.5, 10)
best_threshold = None
best_f1_score = 0

for threshold in thresholds:
    anomalies = reconstruction_error > threshold
    true_labels = subset_df.isna().any(axis=1).astype(int) | (reconstruction_error > threshold).astype(int)

    # Calculate metrics
    report = classification_report(true_labels, anomalies, output_dict=True)
    f1_score = report['1']['f1-score']

    if f1_score > best_f1_score:
        best_f1_score = f1_score
        best_threshold = threshold

print(f"Best Threshold: {best_threshold} with F1 Score: {best_f1_score}")

# Use the best threshold to detect anomalies
anomalies = reconstruction_error > best_threshold
anomalous_data = subset_df[anomalies]

# Additional logic to detect completely missing rows
missing_rows_indicator = subset_df.isna().all(axis=1)
anomalous_data_with_missing_rows = subset_df[missing_rows_indicator | anomalies]

print(f"Anomalies detected:\n{anomalous_data_with_missing_rows}")

# Save anomalies to a file
anomalous_data_with_missing_rows.to_csv('anomalies_with_missing_rows_tune.csv', index=False)
