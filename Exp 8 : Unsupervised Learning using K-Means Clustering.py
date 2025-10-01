import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv('Mall_Customers.csv')
print(data.head())

# Select relevant features for clustering (usually numerical features)
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Added n_init to suppress warning
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of clusters K')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.xticks(K_range)
plt.grid(True)
plt.show()

# From the plot, choose the optimal K (say 5 based on typical result for this dataset)
optimal_k = 5


kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10) # Added n_init to suppress warning
cluster_labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))

# Scatter plot of customers colored by cluster
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=cluster_labels, palette='Set1', s=100, alpha=0.6)

# Mark centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X', label='Centroids')

plt.title(f'K-Means Clustering with K={optimal_k}')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.show()


# Step 6: Analyze Intra-cluster and Inter-cluster distances (Conceptual)
print("""
K-Means tries to minimize the sum of squared distances within each cluster (intra-cluster distance),
which means members of the same cluster are very similar.
At the same time, clusters are well separated to maximize inter-cluster distances.
""")

# Optional: Add cluster labels back to original dataframe
data['Cluster'] = cluster_labels
print(data.groupby('Cluster').mean(numeric_only=True)) # Added numeric_only=True
