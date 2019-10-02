import pandas as pd
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn.neighbors
from matplotlib import style

style.use('ggplot')

spiralData = pd.read_csv('Spiral.csv', delimiter=',')

nObs = spiralData.shape[0]

# Part a

plt.scatter(np.array(spiralData['x']), np.array(spiralData['y']))
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

kvinsp = 2
print("Number of clusters by visual inspection " + str(kvinsp))

# Part b
trainData = spiralData[['x','y']]
kmeans = cluster.KMeans(n_clusters=kvinsp, random_state=60616).fit(trainData)
spiralData['KMClusterLabel'] = kmeans.labels_

plt.scatter(spiralData['x'], spiralData['y'], c = spiralData['KMClusterLabel'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Part c

nNbrs = 3

kNNSpec = sklearn.neighbors.NearestNeighbors(n_neighbors = nNbrs, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(trainData)

# Create the Adjacency and the Degree matrices
Adjacency = np.zeros((nObs, nObs))
Degree = np.zeros((nObs, nObs))

for i in range(nObs):
    for j in i3[i]:
        if (i <= j):
            Adjacency[i,j] = math.exp(- distances[i][j])
            Adjacency[j,i] = Adjacency[i,j]

for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum
        
Lmatrix = Degree - Adjacency

from numpy import linalg as LA
evals, evecs = LA.eigh(Lmatrix)

# Series plot of the smallest ten eigenvalues to determine the number of clusters
plt.scatter(np.arange(0,nNbrs+1,1), evals[0:nNbrs+1,])
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.show()

print("No of nearest neighbors to be used is " + str(nNbrs))

# Part d

sm2evecs = evecs[:,[0,1]]

print("Mean of eigen vector 1 " + str(round(np.mean(sm2evecs, axis=0)[0],10)))
print("Mean of eigen vector 2 " + str(round(np.mean(sm2evecs, axis=0)[1],10)))

print("Standard deviation of eigen vector 1 " + str(round(np.std(sm2evecs, axis=0)[0],10)))
print("Standard deviation of eigen vector 2 " + str(round(np.std(sm2evecs, axis=0)[1],10)))

plt.scatter(sm2evecs[:,0], sm2evecs[:,1])
plt.xlabel('Eigenvector 1')
plt.ylabel('Eigenvector 2')
plt.show()

# Part e

kmeans_spectral = cluster.KMeans(n_clusters=2, random_state=60616).fit(sm2evecs)

spiralData['SpectralClusterKMLabels'] = kmeans_spectral.labels_

plt.scatter(spiralData['x'], spiralData['y'], c = spiralData['SpectralClusterKMLabels'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()