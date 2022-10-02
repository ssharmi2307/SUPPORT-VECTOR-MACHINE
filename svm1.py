# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 20:06:33 2022

@author: Gopinath
"""

import pandas as pd
Train = pd.read_csv('SalaryData_Train(1).csv')
Test = pd.read_csv('SalaryData_Test(1).csv')
Train
Test
Train.info
Test.describe()
Train['Salary'].value_counts()
Test['Salary'].value_counts()

import warnings
warnings.filterwarnings('ignore')



import seaborn as sns
import matplotlib.pyplot as plt
pd.crosstab(Train['occupation'],Train['Salary'])
pd.crosstab(Train['workclass'],Train['Salary'])
pd.crosstab(Train['workclass'],Train['occupation'])
sns.countplot(x='Salary',data= Train)
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()
Train['Salary'].value_counts()

sns.countplot(x='Salary',data= Test)
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()
Test['Salary'].value_counts()

pd.crosstab(Train['Salary'],Train['education']).mean().plot(kind='bar')
pd.crosstab(Train['Salary'],Train['occupation']).mean().plot(kind='bar')
pd.crosstab(Train['Salary'],Train['workclass']).mean().plot(kind='bar')
pd.crosstab(Train['Salary'],Train['sex']).mean().plot(kind='bar')
pd.crosstab(Train['Salary'],Train['relationship']).mean().plot(kind='bar')

string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
##Preprocessing the data. As, there are categorical variables
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
for i in string_columns:
        Train[i]= number.fit_transform(Train[i])
        Test[i]=number.fit_transform(Test[i])
Train
Test

##Capturing the column names which can help in futher process
colnames = Train.columns
colnames
len(colnames)
Train
Test

#target and features

x_train = Train[colnames[0:13]]
y_train = Train[colnames[13]]
x_test = Test[colnames[0:13]]
y_test = Test[colnames[13]]

##Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
x_train = norm_func(x_train)
x_test =  norm_func(x_test)

#SVM Model
from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

#grid search

from sklearn.model_selection import GridSearchCV,KFold
from sklearn import svm
svc = svm.SVC(probability=False)
param={'kernel':['linear','poly','rbf','sigmoid']}
grid=GridSearchCV(svc,param_grid=param,cv=KFold(n_splits=10))
grid.fit(x_train,y_train)
grid.best_score_
grid.best_params_


###final model using kfold
from sklearn.model_selection import cross_val_score
clf_final=SVC(kernel='linear')
result=cross_val_score(clf_final,x_train,y_train,cv=KFold(n_splits=10))
result.mean()
result.min()
result.max()

#####final model using train and test
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred_train = clf.predict(x_train)
y_pred_test  = clf.predict(x_test)

#accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test, y_pred)
cm
accuracy_score(y_train, y_pred_train)
accuracy_score(y_test, y_pred)





