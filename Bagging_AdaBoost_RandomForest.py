# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:57:09 2021

@author: samib
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import GridSearchCV

from sklearn import svm

train_ds, test_ds = np.load('/content/train.npz'), np.load('/content/test.npz')

X = train_ds['arr_0']
Y = train_ds['arr_1']

X_normalized = X/255

X.shape, Y.shape

len(X)
print(X)
print(Y)
X_test = test_ds['arr_0']
X_normalized.shape
print(X_test)
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=35)
X_normalized = preprocessing.normalize(X_train, norm='l2')
clf = LogisticRegressionCV(Cs=10, penalty='l2', dual=False, tol=0.0001, fit_intercept=True, intercept_scaling=1, random_state=42, max_iter=200)
#svc.fit(xtrain, ytrain)
LR = LogisticRegressionCV()
LRparam_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'max_iter': list(range(100,800,100)),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
LR_search = GridSearchCV(LR, param_grid=LRparam_grid, refit = True, verbose = 3, cv=5)

# fitting the model for grid search 
LR_search.fit(X_train, Y_train)
LR_search.best_params_
# summarize
print('Mean Accuracy: %.3f' % LR_search.best_score_)
print('Config: %s' % LR_search.best_params_)
clf.fit(X, Y)
clf.fit(X_train, Y_train)
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
dt.score(X_valid, Y_valid)
clf = AdaBoostClassifier(base_estimator= DecisionTreeClassifier(), n_estimators=2000, learning_rate = 4.0, random_state = 123)
clf.fit(X_train, Y_train)
clf.score(X_valid, Y_valid)
pred = clf.predict(X_test)
df = pd.DataFrame(pred)
df.to_csv('Regression logistique',index=True, index_label='Id', header=['Category'])