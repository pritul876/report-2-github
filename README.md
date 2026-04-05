# report-2-github

Bank Marketing Prediction using Logistic Regression

Objective:
Predict whether a customer will subscribe to a term deposit (yes/no)
based on banking and demographic data.

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data(file_path):
    data = pd.read_csv(file_path, sep=';')
    return data


def preprocess_data(data):
    le = LabelEncoder()

    # Encode categorical columns
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = le.fit_transform(data[column])

    return data


def split_data(data):
    X = data.drop('y', axis=1)
    y = data['y']
    return X, y

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Model Evaluation Results")
    print("=" * 30)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def main():
    # Load dataset
    data = load_data("bank-full.csv")

    # Preprocess data
    data = preprocess_data(data)

    # Split data
    X, y = split_data(data)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

# 8. Run Program

if __name__ == "__main__":
    main()
