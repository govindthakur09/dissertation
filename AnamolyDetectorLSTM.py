import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf  # Import TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split

# Load the datasets
input_file = 'SourceDataSet_AllRecord_FromSource.csv'
subset_file = 'MissingRecord_MissingColumnData.csv'

input_df = pd.read_csv(input_file)
subset_df = pd.read_csv(subset_file)

# Ensure both dataframes have the same columns
assert list(input_df.columns) == list(subset_df.columns), "Input and subset dataframes must have the same columns"

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
input_data, input_scaler = preprocess_data(input_df)

# Define sequence parameters
time_steps = 10  # Number of previous time steps to consider
input_dim = input_data.shape[1]

# Split data into training and validation sets
train_data, val_data = train_test_split(input_data, test_size=0.2, shuffle=False)

# Generate time series data for LSTM
train_generator = TimeseriesGenerator(train_data, train_data, length=time_steps, batch_size=32)
val_generator = TimeseriesGenerator(val_data, val_data, length=time_steps, batch_size=32)

# Define the LSTM model with increased complexity
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(time_steps, input_dim)))
model.add(Dropout(0.3))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=input_dim, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Add EarlyStopping to avoid overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the LSTM model
model.fit(train_generator, epochs=10, shuffle=True, validation_data=val_generator, callbacks=[early_stopping])

# Preprocess subset data using the same scaler
subset_data, _ = preprocess_data(subset_df, scaler=input_scaler)

# Generate time series data for the subset
subset_generator = TimeseriesGenerator(subset_data, subset_data, length=time_steps, batch_size=32)

# Predict using the LSTM model
reconstructed_data = model.predict(subset_generator)

# Calculate the reconstruction error for each sequence
reconstruction_error = []
for i in range(len(subset_generator)):
    seq_true = subset_generator[i][1]  # True sequence
    seq_pred = reconstructed_data[i]   # Predicted sequence
    error = np.mean(np.square(seq_true - seq_pred))  # Use squared error
    reconstruction_error.append(error)

reconstruction_error = np.array(reconstruction_error)

# Set a threshold for anomaly detection based on the 95th percentile
threshold = np.percentile(reconstruction_error, 85)

# Identify anomalies
anomalies = reconstruction_error > threshold
anomalous_data_indices = np.where(anomalies)[0]

# Get the actual anomalous data from the subset
anomalous_data = subset_df.iloc[anomalous_data_indices + time_steps]  # Adjust for time_steps offset

print(f"Anomalies detected:\n{anomalous_data}")

# Save anomalies to a file if needed
anomalous_data.to_csv('anomalies_lstm.csv', index=False)

# Assuming true labels are available; you need to define how to get them
# For this example, we'll create a dummy true_labels array as an example
true_labels = np.zeros(len(reconstruction_error))  # Replace with actual true labels
true_labels[anomalies] = 1  # Assuming anomalies in subset_df are the true anomalies

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

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(true_labels, reconstruction_error)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, lw=2, color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
