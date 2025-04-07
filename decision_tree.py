# -------------------------------------------------------------------------
# AUTHOR: Moaz Ali
# FILENAME: decision_tree.py
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 0.5 hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    for row in data_training:
        refund = 1 if row[0] == 'Yes' else 0
        marital_status = row[1]
        single = 1 if marital_status == 'Single' else 0
        divorced = 1 if marital_status == 'Divorced' else 0
        married = 1 if marital_status == 'Married' else 0
        taxable_income = float(row[2].replace('k', ''))
        X.append([refund, single, divorced, married, taxable_income])

    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    for row in data_training:
        Y.append(1 if row[3] == 'Yes' else 2)

    #loop your training and test tasks 10 times here
    total_accuracy = 0
    for i in range (10):

       #fitting the decision tree to the data by using Gini index and no max_depth
       clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
       clf = clf.fit(X, Y)

       #plotting the decision tree
       tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
       plt.show()

       #read the test data and add this data to data_test NumPy
       df_test = pd.read_csv('cheat_test.csv', sep=',', header=0)
       data_test = np.array(df_test.values)[:,1:]

       correct = 0
       for data in data_test:
           refund = 1 if data[0] == 'Yes' else 0
           marital_status = data[1]
           single = 1 if marital_status == 'Single' else 0
           divorced = 1 if marital_status == 'Divorced' else 0
           married = 1 if marital_status == 'Married' else 0
           taxable_income = float(data[2].replace('k', ''))
           class_predicted = clf.predict([[refund, single, divorced, married, taxable_income]])[0]
           if class_predicted == (1 if data[3] == 'Yes' else 2):
               correct += 1

       accuracy = correct / len(data_test)
       total_accuracy += accuracy

    #print the accuracy of this model during the 10 runs (training and test set).
    print(f'final accuracy when training on {ds}: {total_accuracy / 10}')
