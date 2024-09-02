import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns

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

# Define the VAE model
input_dim = input_data.shape[1]
latent_dim = 14  # Dimensionality of the latent space, can be adjusted
intermediate_dim = 32  # Can be adjusted based on the complexity of your data

# Encoder
inputs = Input(shape=(input_dim,))
h = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# Sampling function for the latent space
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Latent space
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(input_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# Custom loss layer
class VAELossLayer(Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)

    def vae_loss(self, inputs, x_decoded_mean, z_log_var, z_mean):
        reconstruction_loss = K.sum(K.square(inputs - x_decoded_mean), axis=-1)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(reconstruction_loss + kl_loss)

    def call(self, inputs):
        inputs, x_decoded_mean, z_log_var, z_mean = inputs
        loss = self.vae_loss(inputs, x_decoded_mean, z_log_var, z_mean)
        self.add_loss(loss)
        return x_decoded_mean  # Return the output of the decoder

# Define the VAE model with custom loss layer
outputs = VAELossLayer()([inputs, x_decoded_mean, z_log_var, z_mean])
vae = Model(inputs, outputs)

# Compile the model without specifying the loss
vae.compile(optimizer=Adam(learning_rate=0.00001))

# Train the VAE
vae.fit(input_data, input_data, epochs=10, batch_size=50, shuffle=True, validation_split=0.2)

# Preprocess subset data using the same scaler
subset_data, _ = preprocess_data(subset_df, scaler=input_scaler)

# Predict using the VAE
reconstructed_data = vae.predict(subset_data)

# Calculate the reconstruction error
reconstruction_error = np.mean(np.power(subset_data - reconstructed_data, 2), axis=1)

# Set a threshold for anomaly detection
threshold = np.percentile(reconstruction_error, 85)

# Identify anomalies
anomalies = reconstruction_error > threshold
anomalous_data = subset_df[anomalies]

print(f"Anomalies detected:\n{anomalous_data}")

# Save anomalies to a file if needed
anomalous_data.to_csv('anomalies_vae.csv', index=False)

# Assuming true labels are available; you need to define how to get them
# For this example, we'll create a dummy true_labels array as an example
true_labels = np.zeros(len(subset_df))  # Replace with actual true labels
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
