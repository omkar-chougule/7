import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import metrics

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Display dataset information
df.info()

# Display first few rows
df.head()

# Visualize the correlations in the dataset
df.corr().style.background_gradient(cmap='BuGn')

# Drop columns that are not needed
df.drop(['BloodPressure', 'SkinThickness'], axis=1, inplace=True)

# Check for missing values
print(df.isna().sum())

# Describe the dataset
print(df.describe())

# Separate the features (X) and labels (y)
X = df.iloc[:, :df.shape[1] - 1]  # Independent Variables
y = df.iloc[:, -1]                # Dependent Variable
print(X.shape, y.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define KNN function
def knn(X_train, X_test, y_train, y_test, neighbors, power):
    model = KNeighborsClassifier(n_neighbors=neighbors, p=power)
    y_pred = model.fit(X_train, y_train).predict(X_test)
    print(f"Accuracy for K-Nearest Neighbors model: {accuracy_score(y_test, y_pred)}")

    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)     
    
    cr = classification_report(y_test, y_pred)
    print('Classification report:\n', cr)
    

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_neighbors': range(1, 51),
    'p': range(1, 4)
}
grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

# Best model parameters
print(grid.best_estimator_, grid.best_params_, grid.best_score_)

# Apply KNN with the best parameters
knn(X_train, X_test, y_train, y_test, grid.best_params_['n_neighbors'], grid.best_params_['p'])