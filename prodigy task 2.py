import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data_path = r"C:\Users\rouna\Downloads\Mall_Customers.csv"
data = pd.read_csv(data_path)

# Step 2: Check the first few rows and column names to understand the data
print("Columns in the dataset:")
print(data.columns)

# Step 3: Select the features that will be used for clustering
# Assume the dataset contains columns like 'Annual Income (k$)' and 'Spending Score (1-100)'
# Adjust based on the actual columns in your dataset
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Data Preprocessing
# It's a good practice to standardize the features before applying K-means clustering
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 5: Apply K-means Clustering
# Choose the number of clusters (for this example, we will start with 5 clusters)
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# Step 6: Visualize the clusters
plt.figure(figsize=(8,6))
plt.scatter(features['Annual Income (k$)'], features['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.title('Customer Segments based on Spending and Income')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(label='Cluster')
plt.show()

# Step 7: Display cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Optional: You can also print the size of each cluster
print("Size of each cluster:")
print(data['Cluster'].value_counts())
