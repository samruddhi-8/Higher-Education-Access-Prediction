
import csv
import math
import random

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from csv import reader
from math import sqrt
from math import exp
from math import pi
from sklearn.externals import joblib
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

lines = csv.reader(open(r'Higher_education_access_prediction_dataset_sci.csv'))
le = LabelEncoder()  
df = pd.read_csv('Higher_education_access_prediction_dataset_sci.csv')
TestDatadf = pd.read_csv('Higher_education_access_prediction_dataset_sci.csv', sep=",", header=None)
df = DataFrame(TestDatadf)
df = pd.DataFrame(TestDatadf)


df[0]= le.fit_transform(df[0]) 
df[1]= le.fit_transform(df[1])
df[3]= le.fit_transform(df[3])
df[6]= le.fit_transform(df[6])
df[7]= le.fit_transform(df[7])
df[8]= le.fit_transform(df[8])
df[15]= le.fit_transform(df[15])
label = df[15]



dataset = df.values.tolist()
for i in range(len(dataset)):
	dataset[i] = [float(x) for x in dataset[i]]

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.3,random_state=109) 
from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB()


gnb.fit(X_train, y_train)


y_pred = gnb.predict(X_test)
filename1 = 'model.pkl'
joblib.dump(gnb,filename1)


print(le.inverse_transform(y_pred))
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: {}'.format(accuracy_score(y_test, y_pred)))
print('Precision score: {}'.format(precision_score(y_test, y_pred)))
print('Recall score: {}'.format(recall_score(y_test, y_pred)))
print('F1 score: {}'.format(f1_score(y_test, y_pred)))
print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, y_pred))
print("Classification Report")
print(metrics.classification_report(y_test, y_pred))
import pandas
url = "Higher_education_access_prediction_dataset_sci.csv"
names = ['studentaverage', 'gender', 'age', 'studytime', 'familysupport', 'extraclasses','activities','english','biology','physics','chemistry','maths','IT','higher']
data = pandas.read_csv(url, names=names)
pandas.set_option('display.width', 100)
pandas.set_option('precision', 3)
description = data.describe()
print("Statistics of preprocessed data")
print(description)