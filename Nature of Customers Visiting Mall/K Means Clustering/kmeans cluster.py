# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 23:55:41 2020

@author: Nitin
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('E:/Machine Learning/P14-Part4-Clustering/P14-Part4-Clustering/Section 25 - K-Means Clustering/Python/Mall_Customers.csv')
x= dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('elbow method')
plt.xlabel('no of customers')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5,init='k-means++', max_iter=300, n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans == 0,0],x[y_kmeans == 0,1], s=100, c='red', label='careful')
plt.scatter(x[y_kmeans == 1,0],x[y_kmeans == 1,1], s=100, c='blue', label='standard')
plt.scatter(x[y_kmeans == 2,0],x[y_kmeans == 2,1], s=100, c='green', label='target')
plt.scatter(x[y_kmeans == 3,0],x[y_kmeans == 3,1], s=100, c='black', label='careless')
plt.scatter(x[y_kmeans == 4,0],x[y_kmeans == 4,1], s=100, c='orange', label='sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300,c='yellow',label='centroid')
plt.title('clusters of clients')
plt.xlabel('anual income($)')
plt.ylabel('spending score(1-100)')
plt.legend()
plt.show()