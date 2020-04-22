# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 00:48:58 2020

@author: Deepika
"""

import numpy as np                      # needed for arrays and math
import pandas as pd                     # needed to read the data
import matplotlib.pyplot as plt         # used for plotting
from matplotlib import cm as cm         # for the color map
import seaborn as sns                   # data visualization
#from pml53 import plot_decision_regions                # plotting function
import matplotlib.pyplot as plt        # so we can add to plot
from sklearn import datasets            # read the data sets
from sklearn.model_selection import train_test_split   # splits database
from sklearn.preprocessing import StandardScaler       # standardize data
from sklearn.linear_model import Perceptron            # the algorithm
from sklearn.metrics import accuracy_score             # grade the results
from sklearn.linear_model import LogisticRegression    # the algorithm
from sklearn.svm import SVC    
from sklearn.tree import DecisionTreeClassifier         # the algorithm
from sklearn.ensemble import RandomForestClassifier    # the algorithm
from sklearn.neighbors import KNeighborsClassifier     # the algorithm



###Perceptron
def perceptron(X_train, X_test, y_train, y_test):
    
    sc = StandardScaler()                      # create the standard scalar
    sc.fit(X_train)                            # compute the required transformation
    X_train_std = sc.transform(X_train)        # apply to the training data
    X_test_std = sc.transform(X_test)          # and SAME transformation of test data!!!
    
    
    y_train= np.ravel(y_train)
    
    ppn = Perceptron(max_iter=4, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=True)
    ppn.fit(X_train_std, y_train)              # do the training
    #
    
    y_pred = ppn.predict(X_test_std)           # now try with the test data
    shp = y_pred.shape
    
    # Note that this only counts the samples where the predicted value was wrong
    print('Misclassified samples in perceptron: %d' % (y_test.to_numpy().reshape(shp[0],) != y_pred).sum())  # how'd we do?
    print('Accuracy of Perceptron: %.2f' % accuracy_score(y_test.to_numpy().reshape(shp[0],), y_pred))
    
    # vstack puts first array above the second in a vertical stack
    # hstack puts first array to left of the second in a horizontal stack
    # NOTE the double parens!
    
    y_test = np.ravel(y_test)
    
    
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    print('Number in combined ',len(y_combined))

#    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = ppn.predict(X_combined_std)
    print('Misclassified combined samples in perceptron: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy of Perceptron: %.2f' % accuracy_score(y_combined, y_combined_pred))
       
def LR(X_train, X_test, y_train, y_test):
    
    sc = StandardScaler()                       # create the standard scalar
    sc.fit(X_train)                             # compute the required transformation
    X_train_std = sc.transform(X_train)         # apply to the training data
    X_test_std = sc.transform(X_test)           # and SAME transformation of test data!!!
    
    # create logistic regression component.
    # C is the inverse of the regularization strength. Smaller -> stronger!
    #    C is used to penalize extreme parameter weights.
    # solver is the particular algorithm to use
    # multi_class determines how loss is computed - ovr -> binary problem for each label
    y_train= np.ravel(y_train)
    lr = LogisticRegression(C=10, solver='liblinear', multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)                # apply the algorithm to training data
    
    y_pred = lr.predict(X_test_std) 
    shp = y_pred.shape
    
    print('Misclassified samples in Logistic Regression: %d' % (y_test.to_numpy().reshape(shp[0],) != y_pred).sum())  # how'd we do?
    print('Accuracy of Logistic Regression: %.2f' % accuracy_score(y_test.to_numpy().reshape(shp[0],), y_pred))

    # vstack puts first array above the second in a vertical stack
    # hstack puts first array to left of the second in a horizontal stack
    # NOTE the double parens!
    
    y_test = np.ravel(y_test)
    
    
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    print('Number in combined ',len(y_combined))

#    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = lr.predict(X_combined_std)
    print('Misclassified combined samples in Logistic Regression: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy of Logistic Regression: %.2f' % accuracy_score(y_combined, y_combined_pred))
       
def svm(X_train, X_test, y_train, y_test):
    
    sc = StandardScaler()                      # create the standard scalar
    sc.fit(X_train)                            # compute the required transformation
    X_train_std = sc.transform(X_train)        # apply to the training data
    X_test_std = sc.transform(X_test)          # and SAME transformation of test data!!!
    
    y_train= np.ravel(y_train)
    
    svm = SVC(kernel='rbf', tol=1e-3, random_state=0, gamma=.2 , C=10.0, verbose=True)
    svm.fit(X_train_std, y_train)                                  # apply the algorithm

    #
    
    y_pred = svm.predict(X_test_std)           # now try with the test data
    shp = y_pred.shape
    
    # Note that this only counts the samples where the predicted value was wrong
    print('Misclassified samples in SVM: %d' % (y_test.to_numpy().reshape(shp[0],) != y_pred).sum())  # how'd we do?
    print('Accuracy of SVM: %.2f' % accuracy_score(y_test.to_numpy().reshape(shp[0],), y_pred))

    # vstack puts first array above the second in a vertical stack
    # hstack puts first array to left of the second in a horizontal stack
    # NOTE the double parens!
    
    y_test = np.ravel(y_test)
    
    
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    print('Number in combined ',len(y_combined))

#    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = svm.predict(X_combined_std)
    print('Misclassified combined samples in SVM: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy of SVM: %.2f' % accuracy_score(y_combined, y_combined_pred))
       
def dt(X_train, X_test, y_train, y_test):
    
    sc = StandardScaler()                      # create the standard scalar
    sc.fit(X_train)                            # compute the required transformation
    X_train_std = sc.transform(X_train)        # apply to the training data
    X_test_std = sc.transform(X_test)          # and SAME transformation of test data!!!
    
    y_train= np.ravel(y_train)
    tree = DecisionTreeClassifier(criterion='entropy',max_depth=5 ,random_state=0)
    tree.fit(X_train_std,y_train)

    
    y_pred = tree.predict(X_test_std)           # now try with the test data
    shp = y_pred.shape
    
    # Note that this only counts the samples where the predicted value was wrong
    print('Misclassified samples in Decision Tree: %d' % (y_test.to_numpy().reshape(shp[0],) != y_pred).sum())  # how'd we do?
    print('Accuracy of Decision tree: %.2f' % accuracy_score(y_test.to_numpy().reshape(shp[0],), y_pred))
    
    # vstack puts first array above the second in a vertical stack
    # hstack puts first array to left of the second in a horizontal stack
    # NOTE the double parens!
    
    y_test = np.ravel(y_test)
    
    
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    print('Number in combined ',len(y_combined))

#    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = tree.predict(X_combined_std)
    print('Misclassified combined samples in Decision Tree: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy of Decision Tree: %.2f' % accuracy_score(y_combined, y_combined_pred))
       

def rf(X_train, X_test, y_train, y_test):
    
    sc = StandardScaler()                      # create the standard scalar
    sc.fit(X_train)                            # compute the required transformation
    X_train_std = sc.transform(X_train)        # apply to the training data
    X_test_std = sc.transform(X_test)          # and SAME transformation of test data!!!
    
    
    y_train= np.ravel(y_train)
    forest = RandomForestClassifier(criterion='entropy', n_estimators=10,
                                random_state=1, n_jobs=2)
    forest.fit(X_train_std,y_train)

    
    y_pred = forest.predict(X_test_std)           # now try with the test data
    shp = y_pred.shape
    
    # Note that this only counts the samples where the predicted value was wrong
    print('Misclassified samples in Random Forest: %d' % (y_test.to_numpy().reshape(shp[0],) != y_pred).sum())  # how'd we do?
    print('Accuracy of Random Forest: %.2f' % accuracy_score(y_test.to_numpy().reshape(shp[0],), y_pred))

    # vstack puts first array above the second in a vertical stack
    # hstack puts first array to left of the second in a horizontal stack
    # NOTE the double parens!
    
    y_test = np.ravel(y_test)
    
    
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    print('Number in combined ',len(y_combined))

#    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = forest.predict(X_combined_std)
    print('Misclassified combined samples in Random Forest: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy of Random Forest: %.2f' % accuracy_score(y_combined, y_combined_pred))
       
def knn(X_train, X_test, y_train, y_test):
    sc = StandardScaler()                      # create the standard scalar
    sc.fit(X_train)                            # compute the required transformation
    X_train_std = sc.transform(X_train)        # apply to the training data
    X_test_std = sc.transform(X_test)          # and SAME transformation of test data!!!
    
    
    y_train= np.ravel(y_train)
    knn = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
    knn.fit(X_train_std,y_train)

    
    y_pred = knn.predict(X_test_std)           # now try with the test data
    shp = y_pred.shape
    
    # Note that this only counts the samples where the predicted value was wrong
    print('Misclassified samples in K-Nearest neighbour: %d' % (y_test.to_numpy().reshape(shp[0],) != y_pred).sum())  # how'd we do?
    print('Accuracy of K-Nearest neighbour: %.2f' % accuracy_score(y_test.to_numpy().reshape(shp[0],), y_pred))

    # vstack puts first array above the second in a vertical stack
    # hstack puts first array to left of the second in a horizontal stack
    # NOTE the double parens!
    
    y_test = np.ravel(y_test)
    
    
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    print('Number in combined ',len(y_combined))

#    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = knn.predict(X_combined_std)
    print('Misclassified combined samples in K-Nearest neighbour: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy of K-Nearest neighbour: %.2f' % accuracy_score(y_combined, y_combined_pred))
       


data_banknote = pd.read_csv('data_banknote_authentication.txt',sep=",", header=None, names=["variance", "skewness", "curtosis", "entropy", "class"])
#print(data_banknote.head(5))
#print(data_banknote.isnull())

X = data_banknote.iloc[:,0:4]                      # separate the features we want
y = data_banknote.iloc[:,4:5]                            # extract the classifications

# split the problem into train and test
# this will yield 70% training and 30% test
# random_state allows the split to be reproduced
# stratify=y not used in this case
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

perceptron(X_train, X_test, y_train, y_test)
LR(X_train, X_test, y_train, y_test)
svm(X_train, X_test, y_train, y_test)
dt(X_train, X_test, y_train, y_test)
rf(X_train, X_test, y_train, y_test)
knn(X_train, X_test, y_train, y_test)


