# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#data prprocessing

#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= 'NaN', strategy= 'mean', axis = 0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#we need data frame to work with so we do only...x = dataset.iloc[:,:-1]
countries = pd.get_dummies(x.iloc[:,0],drop_first=True)
pd.concat([x,countries],axis=1)


x.drop('Country',axis=1)