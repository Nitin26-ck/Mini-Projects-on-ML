# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:18:24 2020

@author: Nitin
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('E:/Machine Learning/P14-Part4-Clustering/P14-Part4-Clustering/Section 26 - Hierarchical Clustering/Python/Mall_Customers.csv')
x= dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('eucledian distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5 ,linkage='ward')
y_hc = hc.fit_predict(x)

plt.scatter(x[y_hc == 0,0],x[y_hc == 0,1], s=100, c='red', label='careful')
plt.scatter(x[y_hc == 1,0],x[y_hc == 1,1], s=100, c='blue', label='standard')
plt.scatter(x[y_hc == 2,0],x[y_hc == 2,1], s=100, c='green', label='target')
plt.scatter(x[y_hc == 3,0],x[y_hc == 3,1], s=100, c='black', label='careless')
plt.scatter(x[y_hc == 4,0],x[y_hc == 4,1], s=100, c='orange', label='sensible')
plt.title('clusters of clients')
plt.xlabel('annual income($)')
plt.ylabel('spending score(1-100)')
plt.legend()
plt.show()