# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:26:35 2019

@author: Prateek
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Salary_Data.csv')

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3)

from sklearn.linear_model import LinearReression
regressor = LinearRegression()
