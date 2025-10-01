import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('iris.csv', encoding='latin-1')
display(df)
# Define features (X) and target (y)
X = df.drop('species', axis=1)
y = df['species']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create lists to store accuracy scores
accuracy_scores = []
k_values = range(1, 21)

# Iterate through different K values
for k in k_values:
    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the model
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Print the accuracy for each K
for k, accuracy in zip(k_values, accuracy_scores):
    print(f"K = {k}: Accuracy = {accuracy:.4f}")

# Plot the accuracy scores against the K values
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o')
plt.title('KNN Accuracy vs. K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Find the optimal K (the one with the highest accuracy)
optimal_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print(f"Optimal K: {optimal_k}")

# Train the KNN model with the optimal K
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train, y_train)

# Make predictions with the optimal model
y_pred_optimal = knn_optimal.predict(X_test)

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_optimal)
print("\nConfusion Matrix:")
display(conf_matrix)

# Calculate and display the accuracy
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
print(f"\nAccuracy with Optimal K ({optimal_k}): {accuracy_optimal:.4f}")



