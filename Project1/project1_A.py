# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 23:36:14 2020

@author: Deepika
"""

import numpy as np                      # needed for arrays and math
import pandas as pd                     # needed to read the data
import matplotlib.pyplot as plt         # used for plotting
from matplotlib import cm as cm         # for the color map
import seaborn as sns                   # data visualization

## Taken from lecture notes
def mosthighlycorrelated(mydataframe, numtoreport): 
    cormatrix = mydataframe.corr()                      # find the correlations 

    # set the correlations on the diagonal or lower triangle to zero, 
    # so they will not be reported as the highest ones.
    # (The diagonal is always 1; the matrix is symmetric about the diagonal.) 

    # shape returns a tuple, so the * in front of the expression allows the
    # tri function to unpack the tuple into two separate values: rows and cols.

    # The tri function creates an array filled with 1s in the shape of a
    # triangle. If k is 0, then the diagonal and below are all 1s, the rest 0s.
    # If k=-1, the diagonal is 0 but all below the diagonal are 1. If k=-2,
    # then the entries below the diagonal are also 0, and so on. Finally, the
    # transpose is done so that we only keep values above the diagonal.
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T 

    # find the top n correlations 
    #print(cormatrix)
    cormatrix = cormatrix.stack()     # rearrange so the reindex will work...
    #print(cormatrix)

    # Reorder the entries so they go from largest at top to smallest at bottom
    # based on absolute value
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index() 
    #print(cormatrix)

    # assign human-friendly names 
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"] 
    print("\nMost Highly Correlated")
    print(cormatrix.head(numtoreport))     # print the top values

## Taken from lecture notes
def correl_matrix(X):
    print(type(X))
    # create a figure that's 7x7 (inches?) with 100 dots per inch
    fig = plt.figure(figsize=(7,7), dpi=100)

    # add a subplot that has 1 row, 1 column, and is the first subplot
    ax1 = fig.add_subplot(111)

    # get the 'jet' color map
    cmap = cm.get_cmap('jet',30)

    # Perform the correlation and take the absolute value of it. Then map
    # the values to the color map using the "nearest" value
    cax = ax1.imshow(np.abs(X.corr()),interpolation='nearest',cmap=cmap)

    # now set up the axes
    major_ticks = np.arange(0,len(X.columns),1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True,which='both',axis='both')
    plt.title('Correlation Matrix')
    ax1.set_xticklabels(X.columns,fontsize=9)
    ax1.set_yticklabels(X.columns,fontsize=12)

    # add the legend and show the plot
    fig.colorbar(cax, ticks=[-0.4,-0.25,-.1,0,0.1,.25,.5,.75,1])
    plt.show()
    
    ## Covariance Matrix
    print(X.cov())
    

## taken from lecture notes
def pairplotting(df):
    sns.set(style='whitegrid', context='notebook')   # set the apearance
    sns.pairplot(df,height=2.5)                      # create the pair plots
    plt.show()                                       # and show them


# this creates a dataframe similar to a dictionary
# a data frame can be constructed from a dictionary
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
data_banknote = pd.read_csv('data_banknote_authentication.txt',sep=",", header=None, names=["variance", "skewness", "curtosis", "entropy", "class"])
print(data_banknote.head(5))


##  descriptive statistics
print('\nDescriptive Statistics')
print(data_banknote.describe())

mosthighlycorrelated(data_banknote,10)                # generate most highly correlated list
correl_matrix(data_banknote)                         # generate the covariance heat plot
pairplotting(data_banknote) 