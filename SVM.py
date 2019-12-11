# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:29:08 2019

@author: nikhil
"""

import pandas as pd
import numpy as np
import sklearn.svm as svm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

data = pd.read_csv("SpiralWithCluster.csv", delimiter=",")
x = data[['x','y']]
y = data['SpectralCluster']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape='ovr', random_state = 20191108, max_iter = -1)
thisFit = svm_Model.fit(x, y)
y_predictClass = thisFit.predict(x)

x['_PredictedClass_'] = y_predictClass

svm_Mean = x.groupby('_PredictedClass_').mean()
print(svm_Mean)

# Part a

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

# get the separating hyperplane
w = thisFit.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-6, 6)
yy = a * xx - (thisFit.intercept_[0]) / w[1]

#Part b
print('Misclassification rate = ', 1 - metrics.accuracy_score(y, y_predictClass))

#Part c

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(0,2):
    subData = x[x['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.plot(xx, yy, color = 'black', linestyle = ':')
plt.grid(True)
plt.title('Support Vector Machines on Two Segments')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-7,7)
plt.ylim(-7,7)
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

#Part d

x['radius'] = np.sqrt(x['x']**2 + x['y']**2)
x['theta'] = np.arctan2(x['y'], x['x'])

def customArcTan (z):
    theta = np.where(z < 0.0, 2.0*np.pi+z, z)
    return (theta)

x['theta'] = x['theta'].apply(customArcTan)
xTrain = x[['radius','theta']]
x['SpectralCluster'] = y
yTrain = y

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(0,2):
    subData = x[x['SpectralCluster'] == i]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Polar Coordinates Plot')
plt.xlabel('Radius')
plt.ylabel('Angle in Radians')
plt.legend(title = 'Group', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

#Part e

xTrain.loc[3*xTrain['theta'] - 2*xTrain['radius'] > 15, 'Group'] = 0
xTrain.loc[(3*xTrain['theta'] - 2*xTrain['radius'] < 15) & (3*xTrain['theta'] - 8*xTrain['radius'] > -6), 'Group'] = 1
xTrain.loc[(3*xTrain['theta'] - 8*xTrain['radius'] < -6) & (xTrain['theta'] - 2*xTrain['radius'] > -4), 'Group'] = 2
xTrain.loc[xTrain['theta'] - 2*xTrain['radius'] < -4, 'Group'] = 3

carray = ['red', 'blue', 'green', 'black']
plt.figure(figsize=(10,10))
for i in range(0,4):
    subData = xTrain[xTrain['Group'] == i]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Polar Coordinates plot with New Groups')
plt.xlabel('Radius')
plt.ylabel('Angle in Radians')
plt.legend(title = 'Group', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

#Part f

newData = xTrain[['radius','theta', 'Group']]


xx = np.linspace(1, 6)
yy = np.zeros((len(xx),3))
j=0
for grp in [[0,1],[1,2],[2,3]]:
    curData = newData[(newData['Group'] == grp[0]) | (newData['Group'] == grp[1])]
    xnew_Train = curData[['radius','theta']]
    ynew_Train = curData['Group']
    svm_Model_new = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191108, max_iter = -1)
    thisFit_new = svm_Model_new.fit(xnew_Train, ynew_Train) 
    ynew_predictClass = thisFit_new.predict(xnew_Train)
    # get the separating hyperplane
    w = thisFit_new.coef_[0]
    a = -w[0] / w[1]
    yy[:,j] = a * xx - (thisFit_new.intercept_[0] / w[1])
    j+=1
    print("\nThe equation of hyperplane for groups " + str(grp) + " is:")
    print(str(w[0]) + "*x " + str(w[1]) + "*y " + str(thisFit_new.intercept_[0]) + " = 0")

#Part g
carray = ['red', 'green', 'blue', 'black']
plt.figure(figsize=(10,10))
for i in range(0,4):
    subData = newData[newData['Group'] == i]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = i, s = 25)
plt.plot(xx, yy[:,0], color = 'black', linestyle = ':')
plt.plot(xx, yy[:,1], color = 'black', linestyle = ':')
plt.plot(xx, yy[:,2], color = 'black', linestyle = ':')
plt.grid(True)
plt.title('Support Vector Machines on Four Segments')
plt.xlabel('Radius')
plt.ylabel('Angle in Radians')
plt.ylim(-0.5, 6.5)
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

#Part h
h0_xx = xx * np.cos(yy[:,0])
h0_yy = xx * np.sin(yy[:,0])

h1_xx = xx * np.cos(yy[:,1])
h1_yy = xx * np.sin(yy[:,1])

h2_xx = xx * np.cos(yy[:,2])
h2_yy = xx * np.sin(yy[:,2])

# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(0,2):
    subData = x[x['SpectralCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.plot(h0_xx, h0_yy, color = 'green', linestyle = ':')
plt.plot(h1_xx, h1_yy, color = 'blue', linestyle = ':')
plt.plot(h2_xx, h2_yy, color = 'red', linestyle = ':')
plt.grid(True)
plt.title('Support Vector Machines on Two Segments with Hypercurves')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()