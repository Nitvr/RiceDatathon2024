from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
import numpy as np

# Load the data from the uploaded file
file_path = 'training.csv'
data = pd.read_csv(file_path)

cleaned_data = data.dropna(subset=['OilPeakRate'])

# Calculate the percentage of missing values for each column
missing_percentage = cleaned_data.isnull().sum() / len(cleaned_data)

# Identify columns where the percentage of missing values is greater than 80%
columns_to_drop = missing_percentage[missing_percentage >= 0.8].index
print(columns_to_drop)
num_columns_to_drop = len(columns_to_drop)
print(num_columns_to_drop,"are dropped")
# Drop these columns from the DataFrame
col_dropped = cleaned_data.drop(columns_to_drop, axis=1)

row_dropped = col_dropped.dropna()

num_rows, num_columns = row_dropped.shape

row_dropped.drop(['frac_type'], axis=1, inplace=True)



position = row_dropped[['surface_x', 'surface_y', 'bh_x', 'bh_y']]

# Drop rows with NaN values in position DataFrame


# Standardize the data
scaler = StandardScaler()
pos_scaled = scaler.fit_transform(position)

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# Calculating SSE for different number of clusters
sse = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(pos_scaled)
    sse.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(12, 6))  # Increase the figure size
plt.style.use("fivethirtyeight")
plt.plot(range(1, 20), sse)
plt.xticks(range(1, 20))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title("Elbow Method for Optimal k")
plt.show()

kl = KneeLocator(range(1, 20), sse, curve="convex", direction="decreasing")
print('The Elbow point is at cluster:', kl.elbow)

optimal_k = kl.elbow
kmeans = KMeans(n_clusters=optimal_k, **kmeans_kwargs)
# Fit the K-Means model using that K value
kmeans.fit(pos_scaled)
# Assign the cluster centers to the data points
print(len(kmeans.labels_))
row_dropped['cluster'] = kmeans.labels_

plt.figure(figsize=(12, 8))
plt.scatter(pos_scaled[:, 0], pos_scaled[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Scaled Position X')
plt.ylabel('Scaled Position Y')
plt.legend()
print(plt.show())

