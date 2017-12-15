# Ho Ten: Vu The Dung
# MSSV: 14520205
# KNeighborsClassifier
# Last update: 15.12.2017

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split

######Dataset 1
balance_data = pd.read_csv(
'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',sep= ',', header= None)
# print ("Dataset Lenght:: ", len(balance_data))
# print ("Dataset Shape:: ", balance_data.shape)

X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]

######Dataset 2
# iris = datasets.load_iris()
# X = iris.data[:, :3]  #Take the first 3 features.
# Y = iris.target

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_test)
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)
