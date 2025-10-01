import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('weatherHistory.csv').
df.head()

from sklearn.model_selection import train_test_split

X = df.drop(columns=['Precip Type', 'Formatted Date'])  # Drop 'Formatted Date' as well since it's not a feature
y = df['Precip Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from sklearn.tree import DecisionTreeClassifier
# Initialize the Decision Tree Classifier without pruning parameters
dt_classifier = DecisionTreeClassifier(random_state=42)
# Train the classifier on the training data
dt_classifier.fit(X_train, y_train)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5)) # Adjust figure size as needed for better visualization
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=dt_classifier.classes_.tolist())
plt.title("Decision Tree Structure")
plt.show()

from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree Classifier with pruning parameters
# You can experiment with different values for max_depth and min_samples_leaf
dt_classifier_pruned = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)

# Train the pruned classifier on the training data
dt_classifier_pruned.fit(X_train, y_train)

print("Pruned Decision Tree Classifier trained successfully.")

from sklearn.metrics import accuracy_score

# Make predictions on the test set using the unpruned classifier
y_pred_unpruned = dt_classifier.predict(X_test)

# Calculate the accuracy of the unpruned classifier
accuracy_unpruned = accuracy_score(y_test, y_pred_unpruned)
print(f"Accuracy of the unpruned Decision Tree Classifier: {accuracy_unpruned:.4f}")

# Make predictions on the test set using the pruned classifier
y_pred_pruned = dt_classifier_pruned.predict(X_test)

# Calculate the accuracy of the pruned classifier
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
print(f"Accuracy of the pruned Decision Tree Classifier: {accuracy_pruned:.4f}")

# Analyze Feature Importance
import pandas as pd

# Get feature importances from the trained unpruned model
feature_importances = dt_classifier.feature_importances_

# Create a pandas Series for better visualization
feature_importance_series = pd.Series(feature_importances, index=X.columns)

# Sort the features by importance
sorted_feature_importances = feature_importance_series.sort_values(ascending=False)

# Display the sorted feature importances
print("Feature Importances (Unpruned Decision Tree):")
print(sorted_feature_importances)

# Optional: Visualize feature importances
plt.figure(figsize=(10, 6))
sorted_feature_importances.plot(kind='bar')
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
