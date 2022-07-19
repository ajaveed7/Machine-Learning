#SIMPLE LINEAR REGRESSION
#==========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#impoting Dataset

dataset = pd.read_csv(r"E:\Data Science Post Class\Machine Learning\Simple Linear Regressions\SIMPLE LINEAR REGRESSION\Salary_Data.csv")

#Creating independent and dependent variable
X=dataset.iloc[:,:-1].values #independent variable
y=dataset.iloc[:,1].values #dependent variable

#spliting the dataset in training set and testing set
#feature scaling is not mandatory for regression problem
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=0)

#now its time to import our first algorithm Simple Linear Regression
from sklearn.linear_model import LinearRegression
#creatign identifier to hold LinearRegression algorithm
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#model is ready, now think we have X_test and y_test, if we pass X_test to the model and predict Y then we can compare predicted values with y_test actual values

y_pred = regressor.predict(X_test)

# we have predicted value and actual values now we have to check accuracy of the model
# machine learning models wont give 100% accuracy

#visualizing the test result

plt.scatter(X_test,y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience (Test Set)")
plt.show()

# how to see our model is overfitted or underfitted
regressor.score(X_train,y_train) #accuracy is 94.11%

regressor.score(X_test,y_test) # accuracy is 98.82%

