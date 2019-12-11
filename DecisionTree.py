import pandas as pd
import numpy as np
import math
import itertools
from math import log
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

df = pd.read_csv("claim_history.csv", delimiter=",")

clHist_train, clHist_test = train_test_split(df, test_size = 0.3, random_state = 27513, stratify = df['CAR_USE'])

#Part a
print('Number of Observations in Training = ', clHist_train.shape[0])
print("Counts of target variable in Training partition")
print(clHist_train.groupby('CAR_USE').size())
print("\nProportions of target variable in Training partition")
print(clHist_train.groupby('CAR_USE').size() / clHist_train.shape[0])

#Part b
print('\n\nNumber of Observations in Test = ', clHist_test.shape[0])
print("Counts of target variable in Test partition")
print(clHist_test.groupby('CAR_USE').size())
print("\nProportions of target variable in Test partition")
print(clHist_test.groupby('CAR_USE').size() / clHist_test.shape[0])

#Part c
#P(training)
prob_train = 0.7
#P(CarUse = Commercial | Training)
prob_Com_train = (clHist_train.groupby('CAR_USE').size() / clHist_train.shape[0])[0]
#P(CarUse = Private | Training)
prob_Priv_train = (clHist_train.groupby('CAR_USE').size() / clHist_train.shape[0])[1]

#P(test)
prob_test = 0.3
#P(CarUse = Commercial | Test)
prob_Com_test = (clHist_test.groupby('CAR_USE').size() / clHist_test.shape[0])[0]
#P(CarUse = Private | Test)
prob_Priv_test = (clHist_test.groupby('CAR_USE').size() / clHist_test.shape[0])[1]

#P(Training | CarUse = Commercial)
prob_train_Com = (prob_Com_train * prob_train) / (prob_Com_train*prob_train + prob_Com_test*prob_test)
print("\n\nP(Training | CarUse = Commercial) = " + str(prob_train_Com))

#Part d
#P(Test | CarUse = Private)
prob_test_Priv = (prob_Priv_test * prob_test) / (prob_Priv_test*prob_test + prob_Priv_train*prob_train)
print("\n\nP(Test | CarUse = Private) = " + str(prob_test_Priv))


#Question 2

trainData = clHist_train[['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']].dropna()
Y = clHist_train['CAR_USE']

#Part a
rentropy = -1 * ((prob_Com_train)*log(prob_Com_train,2) + (prob_Priv_train)*log(prob_Priv_train,2))
print("\n\nEntropy of root node is " + str(rentropy))

def getRelevantSets(S, varType):
    if varType == "Nominal":
        n = len(S)
        if n==2 or n==3:
            return set(itertools.combinations(S, 1))
        relS=set()
        if n % 2 == 0:
            k=int(n/2)
            for i in range(1,k):
                relS.update(set(itertools.combinations(S, i)))
            kth_subset = set(itertools.combinations(S, k))
            kth_subset = set(itertools.islice(kth_subset, int(len(kth_subset)/2)))
            relS.update(kth_subset)
        else:
            k=math.floor(n/2)
            for i in range(1,k+1):
                relS.update(set(itertools.combinations(S, i)))
        return relS
    elif varType == "Ordinal":
        relL = []
        n=len(S)
        for i in range(1,n):
            relL.append(set(itertools.islice(S, i)))
        relS = set(frozenset(i) for i in relL)
        return [list(x) for x in relS]

def calcEntropy(tot, com, pri):
    p_com = com/tot
    p_pri = pri/tot
    log_p_com = log(p_com,2) if p_com != 0 else 0
    log_p_pri = log(p_pri,2) if p_pri != 0 else 0
    entropy = -1*(p_com * log_p_com + p_pri * log_p_pri)
    return entropy

def getEntropyPerPredictor(data, pred, varType):
    possibleSets = getRelevantSets(set(pred), varType)
    comTot = len(data[data['CAR_USE'] == 'Commercial'])
    priTot = len(data[data['CAR_USE'] == 'Private'])
    nTot = comTot + priTot
    entropyList=[]
    for pset in possibleSets:
        filtData = data[pred.isin(pset)]
        tot = len(filtData)
        com = len(filtData[filtData['CAR_USE'] == 'Commercial'])
        pri = len(filtData[filtData['CAR_USE'] == 'Private'])
        entropy = calcEntropy(tot, com, pri)
        tot2 = nTot - tot
        com2 = comTot - com
        pri2 = priTot - pri
        entropy2 = calcEntropy(tot2, com2, pri2)
        splitEntropy = (tot/nTot)*entropy + (tot2/nTot)*entropy2
        entropyList.append([pset, splitEntropy])
    splitEntropys = np.array(entropyList)[:,1]
    minSplitEntropy = min(splitEntropys)
    return entropyList[np.where(splitEntropys == minSplitEntropy)[0][0]]

def getSplitCondition(node):
    ctype_entropy = getEntropyPerPredictor(node, node['CAR_TYPE'], "Nominal")
    occ_entropy = getEntropyPerPredictor(node, node['OCCUPATION'], "Nominal")
    edu_entropy = getEntropyPerPredictor(node, node['EDUCATION'], "Ordinal")
    minEntPred = [ctype_entropy, occ_entropy, edu_entropy]
    allentropys = np.array(minEntPred)[:,1]
    splitCondition = minEntPred[np.where(allentropys == min(allentropys))[0][0]]
    return splitCondition

#Part b
splitCondition = getSplitCondition(trainData)
print("\n\nSplit condition is " + str(splitCondition[0]))

nodeTrue = trainData[ (trainData['OCCUPATION'] == 'Blue Collar') | (trainData['OCCUPATION'] == 'Student') | 
        (trainData['OCCUPATION'] == 'Unknown') ]

#Removing node true from the training data to get data in other node
nodeFalse = trainData[~trainData.isin(nodeTrue)].dropna()

#Displaying content in node where condition is true
ctypeTrue = set(nodeTrue['CAR_TYPE'])
occTrue = set(nodeTrue['OCCUPATION'])
eduTrue = set(nodeTrue['EDUCATION'])

print("\n\nPredictor name and values in the node where split criterion is true are: ")
print("Car Type: " + str(ctypeTrue))
print("Occupation: " + str(occTrue))
print("Education: " + str(eduTrue))

#Displaying content in node where condition is false
ctypeFalse = set(nodeFalse['CAR_TYPE'])
occFalse = set(nodeFalse['OCCUPATION'])
eduFalse = set(nodeFalse['EDUCATION'])

print("\n\nPredictor name and values in the node where split criterion is false are: ")
print("Car Type: " + str(ctypeFalse))
print("Occupation: " + str(occFalse))
print("Education: " + str(eduFalse))

#Part c
print("\n\nEntropy of split of first layer " + str(splitCondition[1]))

#Part d

#Evaluating nodes for second layer for true node of first layer

splitConditionTrueNode =  getSplitCondition(nodeTrue)

nodeTT = nodeTrue[ (nodeTrue['EDUCATION'] == 'Below High School') ]
nodeTF = nodeTrue[~nodeTrue.isin(nodeTT)].dropna()

splitConditionFalseNode = getSplitCondition(nodeFalse)

nodeFT = nodeFalse[ (nodeFalse['CAR_TYPE'] == 'Minivan') | (nodeFalse['CAR_TYPE'] == 'SUV') | (nodeFalse['CAR_TYPE'] == 'Sports Car')]
nodeFF = nodeFalse[~nodeFalse.isin(nodeFT)].dropna()

print("\n\nNumber of leaves is: 4")

#Part e

#Displaying Counts of target values for leaves
def countTargetVals(node):
    com = len(node[node['CAR_USE'] == 'Commercial'])
    pri = len(node[node['CAR_USE'] == 'Private']) 
    print("Number of values where Car Use is Commercial " + str(com))
    print("Number of values where Car Use is Private " + str(pri))
    return (com/(com+pri))

print("\n\nNode where Layer 1 condition is True and Layer 2 condition is True")
ptt = countTargetVals(nodeTT)

print("\n\nNode where Layer 1 condition is True and Layer 2 condition is False")
ptf = countTargetVals(nodeTF)

print("\n\nNode where Layer 1 condition is False and Layer 2 condition is True")
pft = countTargetVals(nodeFT)

print("\n\nNode where Layer 1 condition is False and Layer 2 condition is False")
pff = countTargetVals(nodeFF)



# Question 3

#Part a

testData = clHist_test[['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']].dropna()
nodeTrueTest = testData[ (testData['OCCUPATION'] == 'Blue Collar') | (testData['OCCUPATION'] == 'Student') | 
        (testData['OCCUPATION'] == 'Unknown') ]
nodeFalseTest = testData[~testData.isin(nodeTrueTest)].dropna()
nodeTTtest = nodeTrueTest[ (nodeTrueTest['EDUCATION'] == 'Below High School') ]
nodeTFtest = nodeTrueTest[~nodeTrueTest.isin(nodeTTtest)].dropna() 
nodeFTtest = nodeFalseTest[ (nodeFalseTest['CAR_TYPE'] == 'Minivan') | (nodeFalseTest['CAR_TYPE'] == 'SUV') | (nodeFalseTest['CAR_TYPE'] == 'Sports Car') ]
nodeFFtest = nodeFalseTest[~nodeFalseTest.isin(nodeFTtest)].dropna()

threshold = float((clHist_train.groupby('CAR_USE').size() / clHist_train.shape[0])['Commercial'])
testData['Predicted_Probability'] = 0
nodeTTtest['Predicted_Probability'] = ptt
nodeTFtest['Predicted_Probability'] = ptf
nodeFTtest['Predicted_Probability'] = pft
nodeFFtest['Predicted_Probability'] = pff

leafNodes = pd.concat([nodeTTtest, nodeTFtest, nodeFTtest, nodeFFtest])
leafNodes.loc[leafNodes['Predicted_Probability'] >= threshold, 'Predicted_Class'] = "Commercial"
leafNodes.loc[leafNodes['Predicted_Probability'] < threshold, 'Predicted_Class'] = "Private"

misClassObs = leafNodes[ leafNodes['CAR_USE'] != leafNodes['Predicted_Class'] ]
misClassRate = len(misClassObs) / len(testData)

print("\n\nMisclassification rate is: " + str(misClassRate))

#Part b
ne = leafNodes[leafNodes['CAR_USE'] == "Commercial"][['CAR_USE','Predicted_Probability']]
nne = leafNodes[leafNodes['CAR_USE'] == "Private"][['CAR_USE','Predicted_Probability']]

ase = ( np.sum(np.square(1-np.array(ne['Predicted_Probability']))) 
+ np.sum(np.square(0-np.array(nne['Predicted_Probability']))) ) / (len(ne) + len(nne))

rase = math.sqrt(ase)
print("\n\nRoot average squared error: " + str(rase))

#Part c
Y_true = 1.0 * np.isin(leafNodes['CAR_USE'], ['Commercial'])
predProbY = np.array(leafNodes['Predicted_Probability'].tolist())
auc = metrics.roc_auc_score(Y_true, predProbY)
print("\n\nArea Under Curve in the Test partition is: " + str(auc))

#Part d

Y = np.array(leafNodes['CAR_USE'].tolist())

# Generate the coordinates for the ROC curve
OneMinusSpecificity, Sensitivity, thresholds = metrics.roc_curve(Y, predProbY, pos_label = 'Commercial')

# Add two dummy coordinates
OneMinusSpecificity = np.append([0], OneMinusSpecificity)
Sensitivity = np.append([0], Sensitivity)

OneMinusSpecificity = np.append(OneMinusSpecificity, [1])
Sensitivity = np.append(Sensitivity, [1])

# Draw the ROC curve
plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.title("Receiver Operating Characteristic curve")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()
