import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

#reading dataset
dataset=pd.read_csv('Mall_Customers.csv')

#Reading the data that could make a cluster
X=dataset.iloc[:,[3,4]].values
# print(X)

#Using elbow method to find minimum no of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
  kmeans=KMeans (n_clusters=i,init='k-means++',random_state=42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

#Visualzing elbow method
import matplotlib.pyplot as plt
# plt.plot(range(1,11),wcss)
# plt.title("The Elbow method ")
# plt.xlabel("Number of cluster")
# plt.ylabel("WCSS")
# plt.show()
#training k means model on the dataset
kmeans=KMeans (n_clusters=5,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(X)

#Visualzing clusters
print(kmeans.cluster_centers_)
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='cluster1')


plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='green',label='cluster2')

plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='blue',label='cluster3')

plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='black',label='cluster4')

plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='brown',label='cluster5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=400,c='cyan',label='Centroid')

plt.title("Customers Clusters")
plt.xlabel("Annual Income k($)")
plt.ylabel("Spending score(1-100)")
plt.legend()
plt.show()