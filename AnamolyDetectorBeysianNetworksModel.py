import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load the datasets
input_file = 'SourceDataSet_AllRecord_FromSource.csv'
subset_file = 'MissingRecord_MissingColumnData.csv'

input_df = pd.read_csv(input_file)
subset_df = pd.read_csv(subset_file)

# Ensure both dataframes have the same columns
assert list(input_df.columns) == list(subset_df.columns), "Input and subset dataframes must have the same columns"

# Preprocess the data: Fill missing values and standardize
def preprocess_data(df, scaler=None, discretizer=None):
    df.fillna(0, inplace=True)  # Fill missing values with 0 or use any other strategy
    if scaler:
        scaled_df = scaler.transform(df)
    else:
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df)

    if discretizer:
        discretized_df = discretizer.transform(scaled_df)
    else:
        discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
        discretized_df = discretizer.fit_transform(scaled_df)

    return discretized_df, scaler, discretizer

# Preprocess input data
input_data, input_scaler, input_discretizer = preprocess_data(input_df)

# Convert the standardized and discretized data back to a DataFrame for Bayesian Network processing
input_data_df = pd.DataFrame(input_data, columns=input_df.columns)

# Create a Bayesian Network structure based on actual column names
columns = input_data_df.columns
model_structure = [(columns[i], columns[i + 1]) for i in range(len(columns) - 1)]
model = BayesianNetwork(model_structure)

# Estimate the parameters using Maximum Likelihood Estimator
model.fit(input_data_df, estimator=MaximumLikelihoodEstimator)

# Perform inference
inference = VariableElimination(model)

# Detect anomalies in the subset data
subset_data, _, _ = preprocess_data(subset_df, scaler=input_scaler, discretizer=input_discretizer)
subset_data_df = pd.DataFrame(subset_data, columns=subset_df.columns)

# Calculate the joint probability of the evidence for each row in the subset data
anomaly_scores = []
for index, row in subset_data_df.iterrows():
    # Calculate the joint probability of the evidence
    joint_prob = 1.0
    for var in model.nodes():
        # Exclude the variable itself from the evidence
        evidence = {k: v for k, v in row.to_dict().items() if k != var}
        prob_dist = inference.query(variables=[var], evidence=evidence)
        joint_prob *= prob_dist.values[0]  # Get the probability for the evidence
    anomaly_scores.append(joint_prob)

# Convert anomaly scores to a numpy array for thresholding
anomaly_scores = np.array(anomaly_scores)

# Set a threshold for anomaly detection (lower joint probability indicates an anomaly)
threshold = np.percentile(anomaly_scores, 85)  # Use a lower percentile for anomaly detection

# Identify anomalies
anomalies = anomaly_scores < threshold
anomalous_data = subset_df[anomalies]

print(f"Anomalies detected:\n{anomalous_data}")

# Save anomalies to a file if needed
anomalous_data.to_csv('anomalies_bn.csv', index=False)

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
fpr, tpr, _ = roc_curve(true_labels, anomaly_scores)
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
precision_vals, recall_vals, _ = precision_recall_curve(true_labels, anomaly_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, lw=2, color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
