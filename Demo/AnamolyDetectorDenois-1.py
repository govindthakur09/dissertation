import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the datasets
#input_file = 'SourceDataSet_AllRecord_FromSource.csv'
#subset_file = 'MissingRecord_MissingColumnData.csv'

#input_file = 'SourceData_For_OutlierDataset.csv'
#subset_file = 'OutlierDataset.csv'

input_file = 'IrrelevantSource_DataSet.csv'
subset_file = 'Relevant_DataSet.csv'

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

# Add noise to the data for training
def add_noise(data, noise_factor=0.1):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=data.shape)
    noisy_data = data + noise
    return np.clip(noisy_data, 0., 1.)  # Ensure values are in the range [0, 1]

noisy_input_data = add_noise(input_data, noise_factor=0.1)

# Define the denoising autoencoder model with more complexity
input_dim = input_data.shape[1]
encoding_dim = 32  # Increased dimension

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
encoder = Dense(encoding_dim // 2, activation="relu")(encoder)  # Additional layer
decoder = Dense(encoding_dim, activation="relu")(encoder)  # Symmetric to encoder
decoder = Dense(input_dim, activation="sigmoid")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the denoising autoencoder with more epochs and smaller batch size
autoencoder.fit(noisy_input_data, input_data, epochs=1000, batch_size=32, shuffle=True, validation_split=0.2)

# Preprocess subset data using the same scaler
subset_data, _ = preprocess_data(subset_df, scaler=input_scaler)

# Predict using the autoencoder
reconstructed_data = autoencoder.predict(subset_data)

# Calculate the reconstruction error
reconstruction_error = np.mean(np.power(subset_data - reconstructed_data, 2), axis=1)

# Set a threshold for anomaly detection using cross-validation or adaptive method
threshold = np.percentile(reconstruction_error, 59)

# Identify anomalies
anomalies = reconstruction_error > threshold
anomalous_data = subset_df[anomalies]



print(f"Anomalies detected:\n{anomalous_data}")

print(f"Number of anomalies detected: {len(anomalous_data)}")

print(f"threshold:\n{threshold}")

print(f"reconstruction_error:\n{reconstruction_error}")

# Save anomalies to a file if needed
anomalous_data.to_csv('denoising_anomalies.csv', index=False)

# Assuming true labels are available; you need to define how to get them
# For this example, we'll create a dummy true_labels array as an example
true_labels = np.zeros(len(subset_df))  # Replace with actual true labels
true_labels[anomalies] = 1  # Assuming anomalies in subset_df are the true anomalies

# Convert anomalies boolean array to integer array
predicted_labels = anomalies.astype(int)

# Calculate performance metrics
precision = precision_score(true_labels, predicted_labels, zero_division=0)
recall = recall_score(true_labels, predicted_labels, zero_division=0)
f1 = f1_score(true_labels, predicted_labels, zero_division=0)
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
