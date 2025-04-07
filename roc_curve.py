# -------------------------------------------------------------------------
# AUTHOR: Moaz Ali
# FILENAME: roc_curve.py
# SPECIFICATION: Train a decision tree and compute ROC curve
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 0.5 hours
# -------------------------------------------------------------------------

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

# read the dataset cheat_data.csv and prepare the data_training numpy array
df = pd.read_csv("cheat_data.csv")

# transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
# be converted to a float.
df["Refund"] = df["Refund"].apply(lambda x: 1 if x == "Yes" else 0)
df["Taxable Income"] = df["Taxable Income"].apply(lambda x: float(x.replace("k", "")))
df = pd.get_dummies(df, columns=["Marital Status"])  # one-hot encode Marital Status

# transform the original training classes to numbers and add them to the vector y. For instance Yes = 1, No = 0, so y = [1, 1, 0, 0, ...]
df["Cheat"] = df["Cheat"].apply(lambda x: 1 if x == "Yes" else 0)

X = df[["Refund", "Marital Status_Divorced", "Marital Status_Married", "Marital Status_Single", "Taxable Income"]].values
y = df["Cheat"].values

# split into train/test sets using 30% for test
trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.3)

# generate random thresholds for a no-skill prediction (random classifier)
ns_probs = [0 for _ in range(len(testy))]

# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)

# keep probabilities for the positive outcome only
dt_probs = dt_probs[:, 1]

# calculate scores by using both classifiers (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()
