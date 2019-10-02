from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

GroceriesDetails = pd.read_csv('Groceries.csv', delimiter=',')

itemListformat = GroceriesDetails.groupby(['Customer'])['Item'].apply(list).values.tolist()

# Calculate the frequency table
nItemPurchase = GroceriesDetails.groupby('Customer').size()
freqTable = pd.Series.sort_index(pd.Series.value_counts(nItemPurchase))
print("Histogram plot of the number of unique items is \n")
plt.bar(freqTable.index.values, freqTable, color='blue', width=1)
plt.xlabel("Number of Unique Items")
plt.ylabel("Frequency")
plt.title("No of Unique Items Histogram")
plt.show()

nTransactions = GroceriesDetails['Customer'].max()
cFreqTable = freqTable.cumsum()
median = cFreqTable[cFreqTable >= nTransactions/2].index[0]
q25 = cFreqTable[cFreqTable >= nTransactions/4].index[0]
q75 = cFreqTable[cFreqTable > 3*nTransactions/4].index[0]

print("25th percentile is " + str(q25))
print("median is " + str(median))
print("75th percentile is " + str(q75))

# Part b

# Convert the Item List format to the Item Indicator format
te = TransactionEncoder()
te_ary = te.fit(itemListformat).transform(itemListformat)
ItemIndicatorformat = pd.DataFrame(te_ary, columns=te.columns_)


largestK = 4
minSup = 75/nTransactions
frequent_itemsets = apriori(ItemIndicatorformat, min_support = minSup, max_len = largestK, use_colnames = True)
print("Number of Itemsets found are " + str(frequent_itemsets.count()['itemsets']))
print("Largest k value among the itemsets is " + str(largestK))

# Part c

assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print("Number of association rules " + str(assoc_rules.count()[0]))

# Part d

plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.title("Support vs Confidence")
plt.show()

# Part e

assoc_rulesC60 = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)
print("Rule 1: \n")
print(str(assoc_rulesC60['antecedents'][0]) + " -> " + str(assoc_rulesC60['consequents'][0]))
print("Support: " + str(assoc_rulesC60['support'][0]))
print("Lift: " + str(assoc_rulesC60['lift'][0]) +"\n")

print("Rule 2: \n")
print(str(assoc_rulesC60['antecedents'][1]) + " -> " + str(assoc_rulesC60['consequents'][1]))
print("Support: " + str(assoc_rulesC60['support'][1]))
print("Lift: " + str(assoc_rulesC60['lift'][1]) +"\n")