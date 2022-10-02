# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 18:54:33 2022

@author: Gopinath
"""

import pandas as pd
df = pd.read_csv("forestfires.csv")
df.shape
df
df.head()

# split as X and Y
Y = df["size_category"]
Y
X = df.iloc[:,2:-1]
X
# standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_scale = SS.fit_transform(X)
X_scale

###############################################################################
#train and test spilt
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size=0.75,random_state=0)  # By default test_size=0.25
X_train.shape
X_test.shape
y_train.shape
y_test.shape

#model creation
from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test  = clf.predict(X_test)

#accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
accuracy_score(y_train, y_pred_train)
accuracy_score(y_test, y_pred_test)
confusion_matrix(y_test, y_pred_test)

#kfold
from sklearn.model_selection import KFold,cross_val_score
clf=SVC()
result=cross_val_score(clf,X,Y,cv=KFold(n_splits=5))
result.std()
result.mean()
result.min()
result.max()

#grid search
from sklearn.svm import SVC
param={'kernel':['linear','poly','rbf','sigmoid']}
from sklearn.model_selection import GridSearchCV
from sklearn import svm
svc = svm.SVC(probability=False)
grid=GridSearchCV(svc,param_grid=param,cv=KFold(n_splits=10))
grid.fit(X,Y)
grid.best_score_
grid.best_params_


###final model using kfold
clf_final=SVC(kernel='linear')
result=cross_val_score(clf_final,X,Y,cv=KFold(n_splits=10))
result.mean()
result.min()
result.max()

#####final model using train and test
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test  = clf.predict(X_test)

#accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
accuracy_score(y_train, y_pred_train)
accuracy_score(y_test, y_pred_test)
confusion_matrix(y_test, y_pred_test)



