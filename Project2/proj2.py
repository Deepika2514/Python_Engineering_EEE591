# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 21:38:34 2020

@author: Deepika
"""

import pandas as pd                                    # data frame
import matplotlib.pyplot as plt                        # modifying plot
from sklearn.model_selection import train_test_split   # splitting data
from sklearn.preprocessing import StandardScaler       # scaling data
from sklearn.decomposition import PCA                  # PCA package
from sklearn.metrics import accuracy_score             # grading
from sklearn.neural_network import MLPClassifier       # Learning algorithm
from sklearn.metrics import confusion_matrix
from warnings import filterwarnings


filterwarnings('ignore')

# Dictionary will contain the data about number of components and accuracy
dic_acc_comp = {}

# read the database. Since it lackets headers, put them in
df_sonar = pd.read_csv('sonar_all_data_2.csv',header = None)


X = df_sonar.iloc[:,0:60].values       # features are in columns 0:60
y = df_sonar.iloc[:,60].values        # classes are in column 60!

# now split the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state = 0)

stdsc = StandardScaler()            # apply standardization
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# NOTE: for now,keep only 2 best features

max_accuracy = float("-inf")

for i in range(1,61):
    pca = PCA(n_components=i,random_state=0)                    # only keep top i  "best" features!
    X_train_pca = pca.fit_transform(X_train_std) # apply to the train data
    X_test_pca = pca.transform(X_test_std)       # do the same to the test data
    
    print(pca.explained_variance_)
    # now create a neural network and train on it
    
    model = MLPClassifier( hidden_layer_sizes = (100,), activation='logistic', max_iter=2000, alpha=0.00001,solver = 'adam', tol=0.0001,random_state = 5)
    #model = MLPClassifier( **clf.best_params_ )
    model.fit(X_train_pca,y_train)
    
    y_pred = model.predict(X_test_pca)              # how do we do on the test data?
    
    print("Number of components:" , i)
    accuracy_score_comp = accuracy_score(y_test, y_pred)
    print('Accuracy: %.2f' % accuracy_score_comp)
    
    dic_acc_comp[i] = accuracy_score_comp
    
    # Printing confusion matrix
    cmat = confusion_matrix(y_test,y_pred)
    #print(cmat)
    
    if accuracy_score_comp > max_accuracy:
        max_accuracy = accuracy_score_comp
        max_confusion_matrix = cmat
        max_component = i
        
        
print("Maximum accuracy is %.2f" % max_accuracy)
print("Number of iterations that achieved the above maximum accuracy is %d"% max_component)

## Plot accuracy vs components
x_coor = [k for k,v in dic_acc_comp.items()]
y_coor = [v for k,v in dic_acc_comp.items()]
plt.plot(x_coor, y_coor) 
  
# naming the x axis 
plt.xlabel('components') 
# naming the y axis 
plt.ylabel('accuracy') 
  
# giving a title to my graph 
plt.title('Graph of components vs accuracy') 
  
# function to show the plot 
plt.show() 

### print the confusion matrix corresponding to the maximum accuracy

print("Confusion matrix corresponding to the maximum accuracy:",max_confusion_matrix)


        
