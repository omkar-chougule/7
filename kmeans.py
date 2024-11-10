import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Load the dataset
df = pd.read_csv('sales_data_sample.csv', encoding='latin-1')

# Drop unnecessary columns
df2 = df.drop(['PRODUCTLINE', 'ORDERDATE', 'STATUS', 'PRODUCTCODE', 'CUSTOMERNAME', 'PHONE',
               'ADDRESSLINE1', 'ADDRESSLINE2', 'CITY', 'STATE', 'POSTALCODE',
               'COUNTRY', 'TERRITORY', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME',
               'DEALSIZE'], axis=1)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df2)

# Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot inertia vs number of clusters
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Calculate the "elbow" point by finding the largest change in the slope
# This finds the difference in inertia between each k and the next, to detect the elbow
slopes = np.diff(inertia)
optimal_k = np.argmin(slopes) + 2  # +2 to correct index offset and account for the drop in slope

# Print the optimal number of clusters
print("Optimal number of clusters:", optimal_k)