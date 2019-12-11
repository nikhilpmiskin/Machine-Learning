# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:46:47 2019

@author: nikhil
"""
import pandas as pd
import sympy
import statsmodels.api as stats
import scipy
import numpy as np

arityData = pd.read_csv("Purchase_Likelihood.csv", delimiter = ",")

Y = arityData["A"].astype('category')

gs = pd.get_dummies(arityData[['group_size']].astype('category'))
hown = pd.get_dummies(arityData[['homeowner']].astype('category'))
mc = pd.get_dummies(arityData[['married_couple']].astype('category'))

model = gs
model = model.join(hown)
model = model.join(mc)

def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pd.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)

def build_mnlogit (fullX, y, debug = 'N'):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print(thisFit.summary())
        print("Model Parameter Estimates:\n", thisParameter)
        print("Model Log-Likelihood Value =", thisLLK)
        print("Number of Free Parameters =", thisDF)

    # Recreat the estimates of the full parameters
    workParams = pd.DataFrame(np.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pd.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)


#Intercept only mode
y=Y
model = pd.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0 = build_mnlogit (model, y, debug = 'N')

#Intercept + Group Size
model = stats.add_constant(gs, prepend=True)
LLK_1GS, DF_1GS, fullParams_1GS = build_mnlogit (model, y, debug = 'N')
testDev = 2 * (LLK_1GS - LLK0)
testDF = DF_1GS - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

# Intercept + Group size + Homeowner
model = gs
model = model.join(hown)
model = stats.add_constant(model, prepend=True)
LLK_1GS_1HOWN, DF_1GS_1HOWN, fullParams_1GS_1HOWN = build_mnlogit (model, y, debug = 'N')
testDev = 2 * (LLK_1GS_1HOWN - LLK_1GS)
testDF = DF_1GS_1HOWN - DF_1GS
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

#Intercept + Group size + Homeowner + Married Couple
model = gs
model = model.join(hown)
model = model.join(mc)
model = stats.add_constant(model, prepend=True)
LLK_1GS_1HOWN_1MC, DF_1GS_1HOWN_1MC, fullParams_1GS_1HOWN_1MC = build_mnlogit (model, y, debug = 'N')
testDev = 2 * (LLK_1GS_1HOWN_1MC - LLK_1GS_1HOWN)
testDF = DF_1GS_1HOWN_1MC - DF_1GS_1HOWN
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

#Intercept + Group size + Homeowner + Married Couple + Group Size * Homeowner
model = gs
model = model.join(hown)
model = model.join(mc)
gs_hown = create_interaction(gs, hown)
model = model.join(gs_hown)
model = stats.add_constant(model, prepend=True)
LLK_GS_HOWN, DF_GS_HOWN, fullParams_GS_HOWN = build_mnlogit (model, y, debug = 'N')
testDev = 2 * (LLK_GS_HOWN - LLK_1GS_1HOWN_1MC)
testDF = DF_GS_HOWN - DF_1GS_1HOWN_1MC
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

#Intercept + Group size + Homeowner + Married Couple + Group Size * Homeowner + Homeowner * Married Couple
model = gs
model = model.join(hown)
model = model.join(mc)
model = model.join(gs_hown)
hown_mc = create_interaction(hown, mc)
model = model.join(hown_mc)
model = stats.add_constant(model, prepend=True)
LLK_HOWN_MC, DF_HOWN_MC, fullParams_HOWN_MC = build_mnlogit (model, y, debug = 'Y')
testDev = 2 * (LLK_HOWN_MC - LLK_GS_HOWN)
testDF = DF_HOWN_MC - DF_GS_HOWN
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

model = gs
model = model.join(hown)
model = model.join(mc)
model = model.join(gs_hown)
model = model.join(hown_mc)

model = stats.add_constant(model, prepend=True)

reduced_form, inds = sympy.Matrix(model.values).rref()

#Part a
print("The aliased parameters found in model are \n")
for i in range(0,len(model.columns)):
    if i not in inds:
        print(model.columns[i])


#Part e
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
gstest = pd.get_dummies(gstest)
howntest = pd.get_dummies(howntest)
mctest = pd.get_dummies(mctest)

testX = gstest
testX = testX.join(howntest)
testX = testX.join(mctest)

gs_howntest = create_interaction(gstest, howntest)
testX = testX.join(gs_howntest)

hown_mctest = create_interaction(howntest, mctest)
testX = testX.join(hown_mctest)
testX = stats.add_constant(testX, prepend=True)

coeff = np.matrix(fullParams_HOWN_MC[['0_y','1_y']])
testXM = np.matrix(testX)

res = np.exp(testXM * coeff)

testX['P(A=1)/P(A=0)'] = res[:,0]
testX['P(A=2)/P(A=0)'] = res[:,1]

pa2=[]
pa0=[]
pa1=[]
for index, row in testX.iterrows():
    pa = 1 / (1 + float(row['P(A=1)/P(A=0)']) + float(row['P(A=2)/P(A=0)']))
    pa0.append(pa)
    pa1.append(row['P(A=1)/P(A=0)'] * pa)
    pa2.append(row['P(A=2)/P(A=0)'] * pa)

pa1pa0=[]
for i in range(0,len(pa0)):
    pa1pa0.append(pa1[i]/pa0[i])

#Part f
print("Max odd value " + str(max(pa1pa0)))    

#Part g
v1 = 1 / float(testX.loc[[2]]['P(A=0)/P(A=2)'])
v2 = 1 / float(testX.loc[[0]]['P(A=0)/P(A=2)'])
ans = v1/v2
print("Required Odds ratio for part g is " + str(ans))

#Part h
v1 = float(testX.loc[[8]]['P(A=0)/P(A=2)']) / float(testX.loc[[8]]['P(A=1)/P(A=2)'])
v2 = float(testX.loc[[0]]['P(A=0)/P(A=2)']) / float(testX.loc[[0]]['P(A=1)/P(A=2)'])
ans = v1/v2
print("Required Odds ratio for part h is " + str(ans))


