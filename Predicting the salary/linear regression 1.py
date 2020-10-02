# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:48:08 2020

@author: Nitin
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('E:/Machine Learning/P14-Part2-Regression/P14-Part2-Regression/Section 6 - Simple Linear Regression/Python/Salary_Data.csv')

x= dataset.iloc[:,:-1].values
y= dataset.iloc[:, 1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3, random_state=0)
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Salary vs experiance')
plt.xlabel('years of experaince')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Salary vs experiance')
plt.xlabel('years of experaince')
plt.ylabel('Salary')
plt.show()