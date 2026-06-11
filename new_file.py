import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_csv("Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
scalar = StandardScaler()
X_scaler = scalar.fit_transform(X)


# inertia_scores = []
# k_range = range(1, 11)
# for k in k_range:
#   K_means = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
#   K_means.fit(X_scaler)
#   inertia_scores.append(K_means.inertia_)


# plt.figure(figsize=(7, 4))
# plt.plot(k_range, inertia_scores, marker='o', c='darkblue', ls='--')
# plt.title("Elboe curve")
# plt.xlabel("Number of clusters")
# plt.ylabel("inertia")
# plt.xticks(k_range)
# plt.tight_layout()
# plt.grid()
# plt.savefig("ElbowCurve.png")

# plt.show()
# plt.close()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
clusterlabel = kmeans.fit_predict(X_scaler)

df['cluster'] = clusterlabel
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green','orange', 'purple']
labels = ["Standard(Average income, average spend)",
          "Target(High income, high spend)",
          "Reckless(Low income, high spend)",
          "Careful(High income, Low spend)",
          "Sensible(Low income, Low spend)"]

for i in range(5):
  plt.scatter(
    X[clusterlabel == i, 0],
    X[clusterlabel == i, 1],
    s = 60,
    c = colors[i],
    label = labels[i],
    alpha=0.8,
    edgecolors= "black"
  )

raw_centroids = scalar.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
  raw_centroids[:,0],
  raw_centroids[:,1],
  s = 250,
  c = "yellow",
  marker='*',
  edgecolors="black",
  linewidths=1.5,
  label= "cluster centroids"
)
plt.title("mall customer market segregation")
plt.xlabel("annual income(k$)")
plt.ylabel("spending score (1-100)")
plt.legend(bbox_to_anchor=(1.05,1), loc = "upper left")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("cluster.png")
plt.show()
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scalar, "mallscaler.pkl")


accuracy = silhouette_score(X_scaler, clusterlabel)
print(accuracy)

print("model saved successfully")