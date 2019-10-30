# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:56:06 2019

@author: Prateek
"""

import numpy as np
import pandas as pd

#read csv file
df = pd.read_csv('USA_Housing.csv')


#deviding into dependent and independent features
x=df.iloc[:,:-2]
y=df.iloc[:,-2]



