import pandas as pd
import numpy as np
data = pd.read_csv('./gg.csv')

dataset=data[['No of users']]
print(dataset)
import random
def init_centroids(k,dataset):
    centroids=[]
    for i in range(0,k):
        point=[]
        for col in dataset.columns:
            point.append(random.uniform(min(dataset[col]),max(dataset[col])))
        centroids.append(point)
    return centroids

import math
def calcdist(dataset,cluster):
    dist = 0
    for idx in range(len(dataset.columns)-1):
        dist += (dataset[dataset.columns[idx]]-cluster[idx])**2
    dist = dist**(1/2)
    return dist
def kmeans(k,dataset):
    centroids = init_centroids(k,dataset)
    dataset['Cluster'] = 0
    original = dataset['Cluster']
    while True:
        dist = pd.Series([math.inf] * len(dataset))
        for idx in range(len(centroids)):
            point = centroids[idx]
            dataset.loc[calcdist(dataset,point)<=dist,['Cluster']] = idx
            dist = pd.concat([dist, calcdist(dataset,point)], axis=1).min(axis=1)
        for idx in range(len(centroids)):
            centroids[idx] =list(dataset[dataset['Cluster']==idx][dataset.columns[0:-1]].mean(axis=0))
        if dataset['Cluster'].eq(original, axis=0).all():
            return dataset,centroids
        else:
            original = dataset['Cluster']


k = int(input('Enter number of clusters : '))
ct = init_centroids(k,dataset)
result,centroids = kmeans(k,dataset)
print(result)



