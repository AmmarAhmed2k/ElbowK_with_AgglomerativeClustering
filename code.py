# -*- coding: utf-8 -*-
"""
Determine optimal number of clusters with dendogram 
Program by Ammar AHmed Siddiqui
#"""

# Importing general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# importing cluster specific library [dendrogram]
import scipy.cluster.hierarchy as sch
# Importing agglomerative clustering methods
from sklearn.cluster import AgglomerativeClustering
# Importing Kmeans from sklearn
from sklearn.cluster import KMeans
# importing silhouette
from sklearn.metrics import silhouette_score

#________________________________________________________________________________________
# Load customers Mall Daataset and necessary extractions
#________________________________________________________________________________________


dataset = pd.read_csv("D:\IMPORTANT\MASTERS\BAHRIA UNIV\Spring2023 Semester\CourseMaterial\Tools in DS\Mall_customers.csv")

# Print dataset top 20 lines only
print(dataset.head())

# Extract 3rd and 4th columns only for analysis
# 3rd Column = Annual Income
# 4th Column = Spending Score

newdata = dataset.iloc[:,[3,4]].values
indexdata = dataset.iloc[:,[0]].values

col3 = dataset.iloc[:,[3]].values
col4 = dataset.iloc[:,[4]].values

#________________________________________________________________________________________
# Code for exploration of K using Elbow Method
#________________________________________________________________________________________


List1 = []

for i in range(1,11): # Range for K Exploration

    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
    # For Model Creation
    kmeans.fit(newdata)
    List1.append(kmeans.inertia_)
    
plt.plot(range(1,11),List1)
plt.title('The ELbow Method')
plt.xlabel('NUmber of Clusters')
plt.ylabel('WCSS')
plt.show()

#sys.exit()

#________________________________________________________________________________________
# Code for exploration of Agglomerative Clustering
#________________________________________________________________________________________

for m in ['ward','average','complete','single']:

    for j in range(2,8): #Actual rnge from 4 to 7
    

        Agg_hc = AgglomerativeClustering(n_clusters = j, affinity = 'euclidean', linkage = m)
        y_hc = Agg_hc.fit_predict(newdata) # model fitting on the dataset
        
        # computing silhouette avg coefficient alongside
        silhouette_avg = silhouette_score(newdata, y_hc)


        mTitle = "Agglomerative: #Clusters = " + str(j)+ " Linkage = "+ m + " Silhoette coeff = " + str(silhouette_avg)
        
        print(mTitle)

        
        plt.scatter(indexdata, y_hc, 1,y_hc)
        plt.title('Values Plot ' + mTitle)
        plt.xlabel('Index')
        plt.ylabel('Cluster #')
        plt.show()

        plt.scatter(col3, col4,10,y_hc)
        plt.title('Scatter Plot ' + mTitle)
        plt.xlabel('Customers - Annual Income')
        plt.ylabel('Customers - Spending Score')
        plt.show()