# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:53:19 2019

@author: nikhil
"""
import pandas as pd
import numpy as np

arityData = pd.read_csv("Purchase_Likelihood.csv", delimiter = ",")

print("Count of target variables: ")
print(arityData.groupby('A').size())

print("\nClass probabilities of target variables: ")
print(arityData.groupby('A').size() / len(arityData))

#Part b

def RowWithColumn (
   rowVar,          # Row variable
   columnVar,       # Column predictor
   show = 'ROW'):   # Show ROW fraction, COLUMN fraction, or BOTH table

   countTable = pd.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
   print("Frequency Table: \n", countTable)
   print( )
   cTotal = countTable.sum(axis = 1)
   rTotal = countTable.sum(axis = 0)
   nTotal = np.sum(rTotal)
   expCount = np.outer(cTotal, (rTotal / nTotal))
   
   chiSqStat = ((countTable - expCount)**2 / expCount).to_numpy().sum()
   
   cV = np.sqrt(chiSqStat/(nTotal * min(len(cTotal)-1, len(rTotal)-1)))
   
   print("Crammer's V statistics is " + str(cV) + "\n")

   if (show == 'ROW' or show == 'BOTH'):
       rowFraction = countTable.div(countTable.sum(1), axis='index')
       print("Row Fraction Table: \n", rowFraction)
       print( )

   if (show == 'COLUMN' or show == 'BOTH'):
       columnFraction = countTable.div(countTable.sum(0), axis='columns')
       print("Column Fraction Table: \n", columnFraction)
       print( )

   return

RowWithColumn(arityData['group_size'],arityData['A'])

#Part c
RowWithColumn(arityData['homeowner'],arityData['A'])

#Part d
RowWithColumn(arityData['married_couple'], arityData['A'])

def categoricalFit(x, targetCol, test):
    
    targetProb = x.groupby(targetCol).size() / len(x)
    
    cols = x.columns.values
    
    d = {}
    
    for col in cols:
        if col == targetCol:
            continue
        countTable = pd.crosstab(index = x[targetCol], columns = x[col], margins = False, dropna = True)
        rowFraction = countTable.div(countTable.sum(1), axis='index')
        d[col] = rowFraction
    
    predProb = {}
    for index, row in test.iterrows():
        psum=0
        for tcat in x[targetCol].unique():
            if tcat not in predProb.keys():
                predProb[tcat] = []
            prob = targetProb[tcat]
            for col in cols:
                if col == targetCol:
                    continue
                rf = d[col]
                cat = row[col]
                p = rf[int(cat)][tcat]
                prob = prob*p
            predProb[tcat].append(prob)
            psum += prob
        for tcat in x[targetCol].unique():
            predProb[tcat][index] = predProb[tcat][index] / psum
    
    for i in predProb.keys():
        test[i] = predProb[i]

gstest = pd.DataFrame(['1','2','3','4',
                       '1','2','3','4',
                       '1','2','3','4',
                       '1','2','3','4'], columns=['group_size'])
howntest = pd.DataFrame(['0','0','0','0',
                         '0','0','0','0',
                         '1','1','1','1',
                         '1','1','1','1'], columns=['homeowner'])
mctest = pd.DataFrame(['0','0','0','0',
                       '1','1','1','1',
                       '0','0','0','0',
                       '1','1','1','1'], columns=['married_couple'])
test = gstest
test = test.join(howntest)
test = test.join(mctest)


categoricalFit(arityData[['group_size', 'homeowner', 'married_couple', 'A']], 'A', test)

pa1 = np.array(test[1])
pa0 = np.array(test[0]) 
pa1pa0 = np.divide(pa1,pa0)
maxIndx = pa1pa0.argmax()

print("Predictor values for maximum odd ratio of P(A=1)/P(A=0): ")
print(test.iloc[maxIndx])    
print("\nMaximum odd value is " + str(max(pa1pa0)))
    
    
    
    
