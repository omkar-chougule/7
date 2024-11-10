import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Load dataset
df = pd.read_csv('uber.csv')

# Clean dataset: Drop unnecessary columns and handle null values
df = df.drop(['Unnamed: 0', 'key'], axis=1)
df.dropna(inplace=True)

# Convert 'pickup_datetime' to datetime and extract useful time features
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
df = df.assign(
    second=df.pickup_datetime.dt.second,
    minute=df.pickup_datetime.dt.minute,
    hour=df.pickup_datetime.dt.hour,
    day=df.pickup_datetime.dt.day,
    month=df.pickup_datetime.dt.month,
    year=df.pickup_datetime.dt.year,
    dayofweek=df.pickup_datetime.dt.dayofweek
)
df = df.drop('pickup_datetime', axis=1)

# Haversine formula to calculate distance between coordinates
def distance_transform(longitude1, latitude1, longitude2, latitude2):
    long1, lati1, long2, lati2 = map(np.radians, [longitude1, latitude1, longitude2, latitude2])
    dist_long = long2 - long1
    dist_lati = lati2 - lati1
    a = np.sin(dist_lati / 2) ** 2 + np.cos(lati1) * np.cos(lati2) * np.sin(dist_long / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a)) * 6371
    return c

# Apply distance calculation
df['Distance'] = distance_transform(
    df['pickup_longitude'], df['pickup_latitude'],
    df['dropoff_longitude'], df['dropoff_latitude']
)

# Drop outliers based on distance and fare_amount
df = df[(df['Distance'] < 60) & (df['fare_amount'] > 0)]
df = df[~((df['fare_amount'] > 100) & (df['Distance'] < 1))]
df = df[~((df['fare_amount'] < 100) & (df['Distance'] > 100))]

# Visualize relationship between distance and fare
plt.scatter(df['Distance'], df['fare_amount'])
plt.xlabel("Distance")
plt.ylabel("Fare Amount")
plt.show()

# Correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='BuGn')
plt.show()

# Standardization of features
X = df['Distance'].values.reshape(-1, 1)
y = df['fare_amount'].values.reshape(-1, 1)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
y_std = scaler.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.2, random_state=0)

# Linear regression model
l_reg = LinearRegression()
l_reg.fit(X_train, y_train)

# Predictions and evaluation
y_pred = l_reg.predict(X_test)
print('Linear Regression Model Performance:')
print(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}")
print(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")
print(f"R Squared (R²): {metrics.r2_score(y_test, y_pred)}")

# Visualization of predictions
plt.scatter(X_test, y_test, c='blue', label='Actual', alpha=0.5, marker='.')
plt.scatter(X_test, y_pred_RF, c='red', label='Predicted', alpha=0.5, marker='.')
plt.xlabel("Distance")
plt.ylabel("Fare Amount")
plt.legend(loc='lower right')
plt.show()