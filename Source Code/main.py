import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# STEP 1: Load Dataset

df = pd.read_csv("Mall_Customers.csv")

# Display Dataset

print("Dataset Preview:")
print(df.head(7))

# STEP 2: Dataset Information

print("\nColumns:")
print(df.columns)

print("\nShape of Dataset:")
print(df.shape)

print("\nNull Values:")
print(df.isnull().sum())

# STEP 3: Select Features

features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print("\nSelected Features:")
print(features.head())

# STEP 4: Scaling

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(features)

print("\nScaled Data:")
print(scaled_data[:7])

# STEP 5: Elbow Method

errors = []

for k in range(2, 13):
    model = KMeans(n_clusters=k, random_state=10)
    model.fit(scaled_data)
    errors.append(model.inertia_)

print("\nError Values:")
print(errors)

# STEP 6: Plot Elbow Curve

plt.figure(figsize=(10,5))

plt.plot(range(2,13), errors, marker='*')

plt.title("Optimal Cluster Detection")
plt.xlabel("K Value")
plt.ylabel("Error")

plt.grid()

plt.show()

# STEP 7: Train Model

model = KMeans(n_clusters=4, random_state=10)

clusters = model.fit_predict(scaled_data)

print("\nCluster Output:")
print(clusters)

# STEP 8: Add Cluster Column

df['Customer_Group'] = clusters

print("\nUpdated Dataset:")
print(df.head())

# STEP 9: Cluster Visualization

plt.figure(figsize=(10,6))

plt.scatter(
    scaled_data[:,0],
    scaled_data[:,1],
    c=clusters,
    cmap='rainbow',
    s=80
)

plt.xlabel("Normalized Income")
plt.ylabel("Normalized Spending")

plt.title("Customer Group Analysis")

plt.show()

# STEP 10: Cluster Centers

plt.figure(figsize=(10,6))

plt.scatter(
    scaled_data[:,0],
    scaled_data[:,1],
    c=clusters,
    cmap='rainbow',
    s=80
)

plt.scatter(
    model.cluster_centers_[:,0],
    model.cluster_centers_[:,1],
    s=400,
    c='black',
    marker='D',
    label='Centers'
)

plt.xlabel("Normalized Income")
plt.ylabel("Normalized Spending")

plt.title("Customer Clusters with Centers")

plt.legend()

plt.show()

# STEP 11: Count Customers

print("\nCustomers in Each Group:")
print(df['Customer_Group'].value_counts())