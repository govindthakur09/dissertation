import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

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

# Define the GAN model components
input_dim = input_data.shape[1]
latent_dim = 14  # Dimension of the latent space, can be adjusted

# Generator model
generator = Sequential([
    Dense(32, activation='relu', input_dim=latent_dim),
    Dense(64, activation='relu'),
    Dense(input_dim, activation='sigmoid')
])

# Discriminator model
discriminator = Sequential([
    Dense(64, activation='relu', input_dim=input_dim),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the discriminator model
discriminator.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# GAN model: stacking generator and discriminator
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = Model(gan_input, gan_output)
gan.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

# Training the GAN
epochs = 10
batch_size = 64

real_labels = np.ones((batch_size, 1))
fake_labels = np.zeros((batch_size, 1))

for epoch in range(epochs):
    # Train the discriminator with real data
    idx = np.random.randint(0, input_data.shape[0], batch_size)
    real_data = input_data[idx]
    d_loss_real = discriminator.train_on_batch(real_data, real_labels)

    # Train the discriminator with generated (fake) data
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_data_batch = generator.predict(noise)
    d_loss_fake = discriminator.train_on_batch(generated_data_batch, fake_labels)

    # Train the generator via GAN model
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, real_labels)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} | Discriminator Loss Real: {d_loss_real[0]} | Discriminator Loss Fake: {d_loss_fake[0]} | Generator Loss: {g_loss}")

# After training, use the discriminator to detect anomalies in the subset data
subset_data, _ = preprocess_data(subset_df, scaler=input_scaler)

# Predict the subset data using the discriminator
discriminator_predictions = discriminator.predict(subset_data)

# Threshold for anomaly detection
threshold = 0.5  # This can be tuned based on your specific use case

# Identify anomalies
anomalies = discriminator_predictions < threshold
anomalous_data = subset_df[anomalies.flatten()]

print(f"Anomalies detected:\n{anomalous_data}")

# Save anomalies to a file if needed
anomalous_data.to_csv('anomalies_gan.csv', index=False)

# Assuming true labels are available; you need to define how to get them
# For this example, we'll create a dummy true_labels array as an example
true_labels = np.zeros(len(subset_df))  # Replace with actual true labels
true_labels[anomalies.flatten()] = 1  # Assuming anomalies in subset_df are the true anomalies

# Convert anomalies boolean array to integer array
predicted_labels = anomalies.astype(int).flatten()

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
fpr, tpr, _ = roc_curve(true_labels, discriminator_predictions.flatten())
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
precision_vals, recall_vals, _ = precision_recall_curve(true_labels, discriminator_predictions.flatten())

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, lw=2, color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
