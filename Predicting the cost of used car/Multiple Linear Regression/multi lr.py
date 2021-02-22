# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:05:50 2021

@author: Nitin
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("E:\ML\CAR DETAILS.csv")

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['name']=encoder.fit_transform(df['name'])
df['year']=encoder.fit_transform(df['year'])
df['fuel']=encoder.fit_transform(df['fuel'])
df['seller_type']=encoder.fit_transform(df['seller_type'])
df['transmission']=encoder.fit_transform(df['transmission'])
df['owner']=encoder.fit_transform(df['owner'])

x=df.loc[:,df.columns!='selling_price']
y=df.iloc[:,2].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))