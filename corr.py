# import pandas as pd


# df = pd.read_csv("Mall_Customers.csv")

# # df["product"] = df['Annual Income (k$)'] +
# print(df.corr(numeric_only=True))


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = [[1,2], [2,1], [8,9], [9,8]]

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

print("Labels:", kmeans.labels_)
print("Centroids:", kmeans.cluster_centers_)