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

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

y_pred = regressor.predict(x_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('km_driven vs selling price')
plt.xlabel('km_driven')
plt.ylabel('selling price')
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,regressor.predict(x_test), color='blue')
plt.title('km_driven vs selling price')
plt.xlabel('km_driven')
plt.ylabel('selling price')
plt.show()

