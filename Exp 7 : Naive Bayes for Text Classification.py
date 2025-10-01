import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('spam.csv', encoding='latin-1')
display(df)

import re
# Convert to lowercase
df['v2'] = df['v2'].str.lower()
# Remove punctuation
df['v2'] = df['v2'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
display(df)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(df['v2'])
display(X.shape)

from sklearn.model_selection import train_test_split
y = df['v1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

from sklearn.naive_bayes import MultinomialNB
# Initialize the Multinomial Naive Bayes model
naive_bayes_model = MultinomialNB()
# Train the model
naive_bayes_model.fit(X_train, y_train)
print("Naive Bayes model trained successfully!")

y_pred = naive_bayes_model.predict(X_test)
print("Predictions made successfully!")

from sklearn.metrics import accuracy_score, classification_report

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train a Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train, y_train)

# Make predictions with the Logistic Regression model
y_pred_lr = logistic_regression_model.predict(X_test)

# Evaluate the Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Compare with Naive Bayes accuracy
print(f"\nNaive Bayes Accuracy: {accuracy:.4f}")
