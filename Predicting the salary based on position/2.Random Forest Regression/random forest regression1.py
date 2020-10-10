# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 19:28:58 2020

@author: Nitin
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('E:/Machine Learning/P14-Part2-Regression/P14-Part2-Regression/Section 11 - Random Forest Regression/Python/Position_Salaries.csv')
x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x,y)

y_pred = regressor.predict([[6.5]])

x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('truth or bluff(random forest regression)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()
