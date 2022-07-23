#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv(r"E:\Data Science Post Class\Machine Learning\Polynomial\25th\1.POLYNOMIAL REGRESSION\Position_Salaries.csv")

X = dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#simple Linear Regression
from sklearn.linear_model  import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(X,y)

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5) #by default it will be two degree, default parameter is called parameter tunning and when we change the degree it is called hyper parametr tunning
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly,y)

#second linear regression with X_poly
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title("Truth or bluff(Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title("Truth or bluff(Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()



#predicting the salary
lin_reg.predict([[6.5]]) #predicting salary with linear model

lin_reg_2.predict(poly_reg.fit_transform([[6.5]])) #predicting salary with poly model

#with poly degree of 5, our model predicts 174878 salary for 6.5 years of experience