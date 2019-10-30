# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:20:37 2019

@author: Prateek
"""

import pandas as pd
import numpy as np

df = pd.read_csv('50_Startups.csv')

#we have 4 coefficient for slops as 4 independent feature and 1 (profit) dependent feature
#state is intersept of slope
#equation of a stright line will defenetly have 1 intercept


#deviding into dependent and independent features
x=df.iloc[:,:-1]
y=df.iloc[:,-1]

#Data preprocessing

states=pd.get_dummies(x['State'],drop_first=True)

x=pd.concat([x,states],axis=1)

x.drop('State',axis=1,inplace=True)


##split into train test
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

##apply for the ML model
from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x_train,y_train)

##prediction
y_pred=regressor.predict(x_test)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)

print(regressor.coef_)


import seaborn as sns

sns.distplot((y_test-y_pred),bins=10)