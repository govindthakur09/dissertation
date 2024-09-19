import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, \
    precision_recall_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
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


# Preprocess data
def preprocess_data(df, scaler=None):
    df.fillna(0, inplace=True)  # Fill missing values with 0
    if scaler:
        scaled_df = scaler.transform(df)
    else:
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df)
    return scaled_df, scaler


input_data, input_scaler = preprocess_data(input_df)

# Define the VAE model
input_dim = input_data.shape[1]
latent_dim = 50  # Adjust latent space
intermediate_dim_1 = 128  # Increased complexity for the first layer
intermediate_dim_2 = 64  # Second hidden layer

# Encoder
inputs = Input(shape=(input_dim,))
h = Dense(intermediate_dim_1, activation='relu')(inputs)
h = BatchNormalization()(h)
h = Dropout(0.3)(h)  # Increased dropout for regularization
h = Dense(intermediate_dim_2, activation='relu')(h)
h = BatchNormalization()(h)
h = Dropout(0.3)(h)

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
decoder_h1 = Dense(intermediate_dim_2, activation='relu')
decoder_h2 = Dense(intermediate_dim_1, activation='relu')
h_decoded = decoder_h1(z)
h_decoded = BatchNormalization()(h_decoded)
h_decoded = Dropout(0.3)(h_decoded)  # Apply dropout after decoder layer
h_decoded = decoder_h2(h_decoded)
h_decoded = BatchNormalization()(h_decoded)

decoder_mean = Dense(input_dim, activation='sigmoid')  # Use sigmoid to constrain outputs to [0, 1]
x_decoded_mean = decoder_mean(h_decoded)


# Custom loss layer
class VAELossLayer(Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)

    def vae_loss(self, inputs, x_decoded_mean, z_log_var, z_mean):
        # Ensure the input and reconstructed data have the same shape
        inputs = K.flatten(inputs)
        x_decoded_mean = K.flatten(x_decoded_mean)
        reconstruction_loss = K.mean(K.binary_crossentropy(inputs, x_decoded_mean))

        # Ensure z_mean and z_log_var are 1D vectors
        kl_loss = 1 + K.flatten(z_log_var) - K.square(K.flatten(z_mean)) - K.exp(K.flatten(z_log_var))
        kl_loss = K.mean(kl_loss) * -0.5

        return reconstruction_loss + kl_loss

    def call(self, inputs):
        inputs, x_decoded_mean, z_log_var, z_mean = inputs
        loss = self.vae_loss(inputs, x_decoded_mean, z_log_var, z_mean)
        self.add_loss(loss)
        return x_decoded_mean


# Define VAE model
outputs = VAELossLayer()([inputs, x_decoded_mean, z_log_var, z_mean])
vae = Model(inputs, outputs)

# Compile the model
vae.compile(optimizer=Adam(learning_rate=0.001))

# Train the VAE
vae.fit(input_data, input_data, epochs=1000, batch_size=32, shuffle=True, validation_split=0.2)

# Preprocess subset data using the same scaler
subset_data, _ = preprocess_data(subset_df, scaler=input_scaler)

# Predict using the VAE
reconstructed_data = vae.predict(subset_data)

# Calculate reconstruction error
reconstruction_error = np.mean(np.power(subset_data - reconstructed_data, 2), axis=1)

# Use percentile threshold for anomaly detection
threshold = np.percentile(reconstruction_error, 60)  # Set threshold at the 95th percentile
anomalies = reconstruction_error > threshold



# Alternatively, use One-Class SVM
# one_class_svm = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
# anomalies = one_class_svm.fit_predict(reconstruction_error.reshape(-1, 1))
# anomalies = anomalies == -1

# Filter anomalies
anomalous_data = subset_df[anomalies]

print(f"Anomalies detected:\n{anomalous_data}")

print(f"threshold:\n{threshold}")

print(f"reconstruction_error:\n{reconstruction_error}")

# Save anomalies to file
anomalous_data.to_csv('anomalies_vae.csv', index=False)

# Create dummy true labels for evaluation
true_labels = np.zeros(len(subset_df))
true_labels[anomalies] = 1

# Convert anomalies to integer array
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

# Confusion Matrix
confusion_mat = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(confusion_mat)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC Curve
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

# Visualize reconstruction error distribution
plt.figure(figsize=(8, 6))
plt.hist(reconstruction_error, bins=50, color='blue', alpha=0.7)
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution")
plt.show()
