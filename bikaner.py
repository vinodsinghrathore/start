# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:29:02 2018

@author: Virender
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:16:30 2018

@author: Virender
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:16:57 2018

@author: Virender
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv("jhodpur.csv" ,delimiter=";")
x=dataset.iloc[:,[6,15]].values

#using elow method optimal no. of cluster
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show( )


#applying k-means to dataset
kmeans=KMeans(n_clusters=3,init="k-means++",max_iter=300,n_init=10,random_state=0)
y_means=kmeans.fit_predict(x)


#visualization
plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,c="red",label="rest day")
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=100,c="blue",label="at 12oclock")
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=100,c="gold",label="at 1and 2pm")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroids')
plt.title("jaipur 2018 tempt graph")
plt.xlabel("temp")
plt.ylabel("solar_rediation")
plt.legend()
plt.show()