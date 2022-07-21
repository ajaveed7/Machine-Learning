#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing Dataset
dataset = pd.read_csv(r"E:\Data Science Post Class\Machine Learning\Simple Linear Regressions\Multiple Linear Regression\50_Startups.csv")

X=dataset.iloc[:,:-1] #independent variable
y=dataset.iloc[:,-1:] #dependent variable

#numerical imputation techniques are not required in regressions.

#we will create dummy variables for charactes i mean sates
X=pd.get_dummies(X)

#splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test) # we predicted y_pred values by passing X_test, this will be helpful to check accuracy

regressor.score(X_train,y_train) #95%
regressor.score(X_test,y_test) #93%

#backward Elimination technique, this will come uder Wrapper filter methods

import statsmodels.formula.api as sm

X = np.append(arr = np.ones((50,1)).astype(int), values = X,axis=1) # in multiple linear regression formula we have y=m1x1+m2x2+m3x3+c,here c is missing so we added c=1.
import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit() # we are doing ordinary least squares here endog is for input and exog is for output

regressor_OLS.summary()
#second index has highest p value hence eliminated this feature, thats why we called this method as backward elimination,
#elimination of attribtute or p value is greater than 0.05 means reject the attribute or eliminate the attribute

X_opt = X[:,[0,1,3,5]]

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit() # we are doing ordinary least squares here endog is for input and exog is for output

regressor_OLS.summary()
#5rd index has highest p value hence rejected
X_opt = X[:,[0,1,3]]

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit() # we are doing ordinary least squares here endog is for input and exog is for output

regressor_OLS.summary()
#3rd object have to eliminate
X_opt = X[:,[0,1]]

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit() # we are doing ordinary least squares here endog is for input and exog is for output

regressor_OLS.summary()
# here 0 is constant if we remove constant then we ended up wiht 1st attribute that is R&D is best for investment 
