import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load Dataset
df = pd.read_csv("1) iris - Copy.csv")

# Save actual species
actual_species = df['species']

# Remove species column
X = df.drop('species', axis=1)

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# Elbow Method
# --------------------------

wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.savefig("elbow_method.png")
plt.show()

# --------------------------
# Train KMeans (K=3)
# --------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

# --------------------------
# Visualize Clusters
# --------------------------

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x='sepal_length',
    y='petal_length',
    hue='Cluster',
    palette='Set1'
)

plt.title("K-Means Clusters")
plt.savefig("clusters.png")
plt.show()

# --------------------------
# Compare with Actual Species
# --------------------------

print("\nCluster vs Species\n")
print(pd.crosstab(df['Cluster'], actual_species))

print("\nTask Completed Successfully!")