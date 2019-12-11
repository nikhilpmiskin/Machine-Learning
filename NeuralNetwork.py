# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:29:47 2019

@author: nikhil
"""

import pandas as pd
import numpy as np
import sklearn.neural_network as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

data = pd.read_csv("SpiralWithCluster.csv", delimiter=",")
x = data[['x','y']]
y = data['SpectralCluster']

threshold = len(data[data["SpectralCluster"] == 1])/len(data)

#Part a
print("Percent of the observations have SpectralCluster equals to 1: " + str(threshold*100) + "%")

#Part b

res=[]
opActFn = 0
for actfn in ["identity", "logistic", "relu", "tanh"]:
    for hlyrs in range(1,6):
        for neurons in range(1,11):
            nnObj = nn.MLPClassifier(hidden_layer_sizes = (neurons,)*hlyrs,
                                 activation = actfn, verbose = False,
                                 solver = 'lbfgs', learning_rate_init = 0.1,
                                 max_iter = 5000, random_state = 20191108)
            thisFit = nnObj.fit(x, y)
            y_predProb = nnObj.predict_proba(x)
            y_pred = np.where(y_predProb[:,1] >= threshold, 1, 0)
            mlp_loss = nnObj.loss_
            mlp_misclrt = 1 - metrics.accuracy_score(y, y_pred)
            opActFn = nnObj.out_activation_
            res.append([actfn, hlyrs, neurons, nnObj.n_iter_, mlp_loss, mlp_misclrt])
result = pd.DataFrame(res)
result.columns = ["Activation function", "Number of layers", "Number of neurons per layer", "Number of iterations performed", "Loss value", "Misclassification rate"]

bestModels = []
for actfn in ["identity", "logistic", "relu", "tanh"]:
    modelPerf = result[result['Activation function'] == actfn]
    bestModel = modelPerf[modelPerf['Loss value'] == min(modelPerf['Loss value'])]
    bestModels.append(bestModel.values[0].tolist())

bestRes = pd.DataFrame(bestModels)
bestRes.columns = ["Activation function", "Number of layers", "Number of neurons per layer", "Number of iterations performed", "Loss value", "Misclassification rate"]

#Part c
print("\nActivation function for the output layer is " + opActFn)

#Part d

leastLoss = result[result['Loss value'] == min(result['Loss value'])]
print("\nThe details of the lowest loss and lowest misclassification rate are:")
print(leastLoss.iloc[0])

#Part e
nnObj = nn.MLPClassifier(hidden_layer_sizes = (int(leastLoss['Number of neurons per layer']),)*int(leastLoss['Number of layers']),
                                 activation = leastLoss['Activation function'].values[0], verbose = False,
                                 solver = 'lbfgs', learning_rate_init = 0.1,
                                 max_iter = 5000, random_state = 20191108)
thisFit = nnObj.fit(x, y)
y_predProb = nnObj.predict_proba(x)
y_pred = np.where(y_predProb[:,1] >= threshold, 1, 0)
x['_PredictedClass_'] = y_pred
mlp_Mean = x.groupby('_PredictedClass_').mean()

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = x[x['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('MLP (' + str(leastLoss['Number of layers'].values[0]) + ' Layers, ' + str(leastLoss['Number of neurons per layer'].values[0]) + ' Neurons) ' + 'Activation function: '+str(leastLoss['Activation function'].iloc[0]))
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

#Part e
data["Pred Prob(SpectralCluster = 1)"] = y_predProb[:,1]
g = data.groupby("SpectralCluster")


print("\nThe required values for SpectralCluster = 0 are: ")
print(g.get_group(0).describe())

print("\nThe required values for SpectralCluster = 1 are: ")
print(g.get_group(1).describe())
