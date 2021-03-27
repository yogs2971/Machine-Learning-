# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 21:32:05 2021

@author: yogsg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")


#divide in to dependent and independent variable

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


#Taking care of missing data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

imputer.fit(x[:,1:3])

x[:,1:3] =imputer.transform(x[:,1:3])

# Encoding categorical data

# Encoding the independent variable

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')

x = np.array(ct.fit_transform(x))


#Encoding the Dependent variable

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)


#Splitting dataset into training and test set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=1)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test [:,3:] = sc.transform(X_test[:,3:])

print(X_train)
print("\n",X_test)
