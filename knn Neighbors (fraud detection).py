# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 10:52:28 2019

@author: nikhil
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.neighbors import KNeighborsClassifier as kNC

print("------------Question 3------------\n\n")

csvFile = pd.read_csv('Fraud.csv')
fraudsInfo = csvFile['FRAUD']
sum = 0
frauds1 = fraudsInfo.sum()
fraudPercent = frauds1/len(fraudsInfo)
print("Percent of fraudulent investigations is " + str(round(fraudPercent,4)))

def plotDataForVar(csvFile, var):
    vf1 = csvFile[csvFile['FRAUD'] == 1][var].values.copy()
    vf0 = csvFile[csvFile['FRAUD'] == 0][var].values.copy()
    data = [vf0, vf1]
    plt.boxplot(data, vert=False)
    plt.title(var)
    plt.yticks([1, 2], ["Non Fraudulent", "Fraudulent"])
    plt.show()

plotDataForVar(csvFile, 'TOTAL_SPEND')
plotDataForVar(csvFile, 'DOCTOR_VISITS')
plotDataForVar(csvFile, 'NUM_CLAIMS')
plotDataForVar(csvFile, 'MEMBER_DURATION')
plotDataForVar(csvFile, 'OPTOM_PRESC')
plotDataForVar(csvFile, 'NUM_MEMBERS')

ts = csvFile['TOTAL_SPEND']
dv = csvFile['DOCTOR_VISITS']
nc = csvFile['NUM_CLAIMS']
md = csvFile['MEMBER_DURATION']
op = csvFile['OPTOM_PRESC']
nm = csvFile['NUM_MEMBERS']

x = np.matrix([ts, dv, nc, md, op, nm]).transpose()


xtx = x.transpose() * x
print("t(x) * x = \n", xtx)

# Eigenvalue decomposition
evals, evecs = LA.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("Eigenvectors of x = \n",evecs)

filt = []
for i in range(0,len(evals)):
    if evals[i] >= 1:
        filt.append(i)



evecsfilt = evecs[filt].transpose()
evalsfilt = evals[filt] 

print("Number of Dimesions used is " + str(len(evalsfilt)) + "\n")

# Here is the transformation matrix
transf = evecs * LA.inv(np.sqrt(np.diagflat(evals)))
print("Transformation Matrix = \n", transf)

# Here is the transformed X
transf_x = x * transf
print("The Transformed x = \n", transf_x)

# Check columns of transformed X
xtx1 = transf_x.transpose() * transf_x
print("Expect an Identity Matrix = \n", xtx1)

# Specify the kNN
kNNSpec1 = kNN(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec1.fit(x)

kNNSpec = kNC(n_neighbors = 5)
nbrsC = kNNSpec.fit(x,np.array(fraudsInfo.values))
scor = kNNSpec.score(x, np.array(fraudsInfo.values), sample_weight=None)

focal = [7500, 15, 3, 127, 2, 2]
transfFocal = focal*transf
myNeighbors_t = nbrs.kneighbors(transfFocal, return_distance = False)

myNeighbors_t_values = x[myNeighbors_t]

yac = np.matrix(fraudsInfo.values).transpose()
targetVals = yac[myNeighbors_t]

class_prob = nbrsC.predict_proba(x)
