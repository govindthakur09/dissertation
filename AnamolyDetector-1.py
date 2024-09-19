import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns

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
assert set(input_df.columns) == set(subset_df.columns), "Input and subset dataframes must have the same columns"

# Align both DataFrames by merging based on 'card_number' or another unique identifier
merged_df = pd.merge(input_df, subset_df, on='card_number', how='inner', suffixes=('_input', '_subset'))

# Extract aligned DataFrames
aligned_input_df = merged_df[[col for col in merged_df.columns if col.endswith('_input')]].rename(columns=lambda x: x.rstrip('_input'))
aligned_subset_df = merged_df[[col for col in merged_df.columns if col.endswith('_subset')]].rename(columns=lambda x: x.rstrip('_subset'))

# Preprocess the data: Fill missing values and standardize
def preprocess_data(df, scaler=None):
    df.fillna(0, inplace=True)  # Fill missing values with 0
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
encoding_dim = 256  # Increased dimensionality of the encoding layer

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activity_regularizer=l2(0.01))(input_layer)
encoder = LeakyReLU(alpha=0.2)(encoder)
encoder = BatchNormalization()(encoder)
encoder = Dense(encoding_dim // 2, activity_regularizer=l2(0.01))(encoder)
encoder = LeakyReLU(alpha=0.2)(encoder)
encoder = BatchNormalization()(encoder)
encoder = Dropout(0.3)(encoder)

decoder = Dense(encoding_dim // 2)(encoder)
decoder = LeakyReLU(alpha=0.2)(decoder)
decoder = BatchNormalization()(decoder)
decoder = Dense(input_dim, activation="sigmoid")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the autoencoder with increased epochs and batch size
history = autoencoder.fit(input_data, input_data, epochs=1000, batch_size=32, shuffle=True, validation_split=0.2)

# Preprocess subset data using the same scaler
subset_data, _ = preprocess_data(aligned_subset_df, scaler=input_scaler)

# Predict using the autoencoder
reconstructed_data = autoencoder.predict(subset_data)

# Calculate the reconstruction error
reconstruction_error = np.mean(np.abs(subset_data - reconstructed_data), axis=1)

# Optimize threshold dynamically using Precision-Recall Curve
precision_vals, recall_vals, thresholds = precision_recall_curve((aligned_input_df.values != aligned_subset_df.values).any(axis=1), reconstruction_error)
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals)
optimal_threshold = thresholds[np.argmax(f1_scores)]



# Identify anomalies using the optimal threshold
anomalies = reconstruction_error > optimal_threshold
# Add card_number to anomalous data
anomalous_data = aligned_subset_df[anomalies].copy()
anomalous_data['card_number'] = merged_df['card_number'][anomalies]

# Print the number of anomalies detected and save to CSV
print(f"Number of anomalies detected: {len(anomalous_data)}")
print(f"Anomalies detected:\n{anomalous_data}")

print(f"threshold:\n{optimal_threshold}")

print(f"reconstruction_error:\n{reconstruction_error}")

# Save anomalies along with card_number to a file
anomalous_data.to_csv('anomalies_with_card_number.csv', index=False)

# Performance metrics
true_labels = (aligned_input_df.values != aligned_subset_df.values).any(axis=1).astype(int)
predicted_labels = anomalies.astype(int)

precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Confusion matrix
confusion_mat = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(true_labels, reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve
plt.plot(recall_vals, precision_vals, lw=2, color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
