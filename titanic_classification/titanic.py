# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:10:26 2019

@author: ROTIMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import csv

train_csv = pd.read_csv('trainTitanic.csv')
final_csv = pd.read_csv('testTitanic.csv')

dataset = train_csv

test_dataset = final_csv

sex_transform = []
for sex in dataset.Sex:
    if sex == 'male':
        sex_transform.append(1)
    else:
        sex_transform.append(2)
dataset['sex_transform'] = sex_transform
numeric_features = dataset.select_dtypes(include=[np.number])
numeric_features = numeric_features.drop(['PassengerId'], axis=1)
numeric_features = numeric_features.select_dtypes(include=[np.number]).interpolate().dropna()
numeric_features = numeric_features[numeric_features['Fare']<300]
#numeric_features = numeric_features[numeric_features['Age']<70]
Y = numeric_features['Survived']
X = numeric_features.drop('Survived', axis=1)


#plt.plot(X.Age[Y==1], 'ro')
##plt.plot(X.Age[Y==0], X.Pclass[Y==0], 'go')
#plt.xlabel('age')
#plt.ylabel('fare')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.02, random_state=42)

clf = clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#plt.plot(X_train["sepal.length"], X_train["sepal.width"])
print(accuracy_score(y_test, y_pred))

test_data = test_dataset
test_sex_transform = []
sex_transform = []
for sex in test_data.Sex:
    if sex == 'male':
        test_sex_transform.append(1)
    else:
        test_sex_transform.append(2)
        
test_data['sex_transform'] = test_sex_transform
test_data = test_data.select_dtypes(include=[np.number]).interpolate().drop('PassengerId', axis=1)
print(test_data.head())
y_test_pred = clf.predict(test_data)
print(len(y_test_pred))

append_list = [['PassengerId', 'Survived']]

for x,y in zip(list(test_dataset.PassengerId), list(y_test_pred)):
    append_list.append([x, y])


with open('submit.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(append_list)

csvFile.close()
