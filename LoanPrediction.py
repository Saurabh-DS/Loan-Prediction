# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df = pd.read_csv("train_ctrUa4K.csv")

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mode()[0], inplace=True)

df['Gender']= df['Gender'].replace('Male', '1')
df['Gender']= df['Gender'].replace('Female', '0')

df['Education']= df['Education'].replace('Graduate', '1')
df['Education']= df['Education'].replace('Not Graduate', '0')

df['Self_Employed']= df['Self_Employed'].replace('Yes', '1')
df['Self_Employed']= df['Self_Employed'].replace('No', '0')

df['Married']= df['Married'].replace('Yes', '1')
df['Married']= df['Married'].replace('No', '0')

df['Property_Area']= df['Property_Area'].replace('Urban', '1')
df['Property_Area']= df['Property_Area'].replace('Semiurban', '2')
df['Property_Area']= df['Property_Area'].replace('Rural', '3')

df['Loan_Status']= df['Loan_Status'].replace('Y', '1')
df['Loan_Status']= df['Loan_Status'].replace('N', '0')

df['Dependents'] = df['Dependents'].str.strip("+")

df[['Gender']] = df[['Gender']].astype(float)
df[['Married']] = df[['Married']].astype(float)
df[['Dependents']] = df[['Dependents']].astype(float)
df[['Education']] = df[['Education']].astype(float)
df[['Self_Employed']] = df[['Self_Employed']].astype(float)
df[['Property_Area']] = df[['Property_Area']].astype(float)
df[['Loan_Status']] = df[['Loan_Status']].astype(float)
df[['ApplicantIncome']] = df[['ApplicantIncome']].astype(float)

X = df[['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]
y = df['Loan_Status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=500)
model.fit(X_train,y_train)

LR = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,LR)

inputt=[int(x) for x in "1 1 1 1 0 4000 2000 130 360 1 2".split(' ')]
final=[np.array(inputt)]

p = model.predict(final)

import pickle
pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))