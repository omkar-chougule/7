import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Load the dataset
df = pd.read_csv('emails.csv')
df.info()  # To check dataset structure

# Drop 'Email No.' column and clean the data
df.drop(columns=['Email No.'], inplace=True)
df.isna().sum()  # Check for missing values
df.describe()  # Descriptive statistics

# Separate features and labels
X = df.iloc[:, :-1]  # Independent variables (features)
y = df.iloc[:, -1]   # Dependent variable (labels)

# Split the dataset into training and test sets (85% train, 15% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=8)

# Machine Learning models
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=2),
    "Linear SVM": LinearSVC(random_state=8, max_iter=900000),
    "Polynomial SVM": SVC(kernel="poly", degree=2, random_state=8),
    "RBF SVM": SVC(kernel="rbf", random_state=8),
    "Sigmoid SVM": SVC(kernel="sigmoid", random_state=8)
}

# Train and evaluate each model
for model_name, model in models.items():
    y_pred = model.fit(X_train, y_train).predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name} model: {accuracy}")

print(metrics.classification_report(y_test, y_pred))

metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)     