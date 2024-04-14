import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data=pd.read_csv("Mall_Customers.csv")
print(data)

features=['Annual Income (k$)','Spending Score (1-100)']
X=data[features]

scaler=StandardScaler()
scaled_data=scaler.fit_transform(X)

em_model=GaussianMixture(n_components=5,random_state=42)
em_clusters=em_model.fit_predict(scaled_data)

kmeans_model=KMeans(n_clusters=5,random_state=42)
kmeans_clusters=kmeans_model.fit_predict(scaled_data)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.scatter(X.iloc[:, 0],X.iloc[:, 1],c=em_clusters,cmap='viridis')
plt.title("Em Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.subplot(1,2,2)
plt.scatter(X.iloc[:, 0],X.iloc[:, 1],c=kmeans_clusters,cmap='viridis')
plt.title("K_means Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.tight_layout()
plt.show()

