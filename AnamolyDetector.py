import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
input_file = 'SourceDataSet_AllRecord_FromSource.csv'
subset_file = 'MissingRecord_MissingColumnData.csv'

input_df = pd.read_csv(input_file)
subset_df = pd.read_csv(subset_file)

# Ensure both dataframes have the same columns
assert set(input_df.columns) == set(subset_df.columns), "Input and subset dataframes must have the same columns"

# Align both DataFrames by merging based on 'card_number' or another unique identifier
merged_df = pd.merge(input_df, subset_df, on='card_number', how='inner', suffixes=('_input', '_subset'))

# Extract aligned DataFrames
aligned_input_df = merged_df[[col for col in merged_df.columns if col.endswith('_input')]].rename(columns=lambda x: x.rstrip('_input'))
aligned_subset_df = merged_df[[col for col in merged_df.columns if col.endswith('_subset')]].rename(columns=lambda x: x.rstrip('_subset'))

# Preprocess the data: Fill missing values and standardize
def preprocess_data(df, scaler=None):
    df.fillna(0, inplace=True)  # Fill missing values with 0 or use any other strategy
    if scaler:
        scaled_df = scaler.transform(df)
    else:
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df)
    return scaled_df, scaler

# Preprocess input data
input_data, input_scaler = preprocess_data(aligned_input_df)

# Define the autoencoder model
input_dim = input_data.shape[1]
encoding_dim = 25  # Can be adjusted

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.1), loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(input_data, input_data, epochs=100, batch_size=50, shuffle=True, validation_split=0.2)

# Preprocess subset data using the same scaler
subset_data, _ = preprocess_data(aligned_subset_df, scaler=input_scaler)

# Predict using the autoencoder
reconstructed_data = autoencoder.predict(subset_data)

# Calculate the reconstruction error
reconstruction_error = np.mean(np.power(subset_data - reconstructed_data, 2), axis=1)

# Set a threshold for anomaly detection
threshold = np.percentile(reconstruction_error, 85)

# Identify anomalies
anomalies = reconstruction_error > threshold
anomalous_data = aligned_subset_df[anomalies]

print(f"Number of anomalies detected: {len(anomalous_data)}")
print(f"Anomalies detected:\n{anomalous_data}")

# Save anomalies to a file if needed
anomalous_data.to_csv('anomalies.csv', index=False)

# Additional analysis to help understand the performance
print(f"Reconstruction error threshold: {threshold:.4f}")
print(f"Minimum reconstruction error: {np.min(reconstruction_error):.4f}")
print(f"Maximum reconstruction error: {np.max(reconstruction_error):.4f}")

print(f"Number of anomalies detected: {len(anomalous_data)}")

# Plotting reconstruction error distribution
plt.hist(reconstruction_error, bins=50, alpha=0.75)
plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=1)
plt.title('Distribution of Reconstruction Errors')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.show()

# Create true labels based on the differences between aligned_input_df and aligned_subset_df
true_labels = (aligned_input_df.values != aligned_subset_df.values).any(axis=1).astype(int)

# Convert anomalies boolean array to integer array
predicted_labels = anomalies.astype(int)

# Calculate performance metrics
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Optionally print confusion matrix
confusion_mat = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(confusion_mat)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(true_labels, reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plot Precision-Recall curve
from sklearn.metrics import precision_recall_curve

precision_vals, recall_vals, _ = precision_recall_curve(true_labels, reconstruction_error)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, lw=2, color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
