# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:21:42 2020

@author: Nitin
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('E:/Machine Learning/P14-Part2-Regression/P14-Part2-Regression/Section 8 - Polynomial Regression/Python/Position_Salaries.csv')
x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values



from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('truth or bluff(linear regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='blue')
plt.title('truth or bluff(polynomial regression)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()

y_pred = lin_reg.predict([[6.5]])

y_pred2 = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
