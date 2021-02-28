import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"E:\ML\CAR DETAILS.csv")
df.head(5)
df.info()
df.describe()

df1=pd.read_csv(r"E:\ML\CAR DETAILS.csv").drop(['name','year','fuel','seller_type','transmission','owner'],axis=1)
x= df1.iloc[:,:-1].values
y= df1.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('car details(poly regression)')
plt.xlabel('km_driven')
plt.ylabel('selling price')
plt.show()

x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='blue')
plt.title('car details(polynomial regression)')
plt.xlabel('km_driven')
plt.ylabel('selling price')
plt.show()