#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"E:\Data Science Post Class\Kaggle\Multiple Linear Regression\kc_house_data.csv")

#we dont need id and date so removed this attributes
dataset.drop(['id','date'], axis=1, inplace=True)

dataset.isnull().sum() #no null vaues or missing values in the datset

X=dataset.iloc[:,1:] #independent variable

y=dataset.iloc[:,:1] #dependent variable

#spliting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

#importing linear regression model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)

regressor.score(X_train,y_train) #70.05% training accuracy
regressor.score(X_test,y_test) #69.49% testing accuracy

#Backward elimination
import statsmodels.api as sm
X = np.append(arr=np.ones((21613,1)).astype(int),values=X,axis=1)#added one as a constant in the dataset

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#removeing 5 as the p-value is greater than 0.05
X_opt = X[:,[0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]]

regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()