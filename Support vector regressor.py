import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"E:\Data Science Post Class\Machine Learning\Support Vector Regression\Position_Salaries.csv")

#dependent and indepencent variable
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel='poly', degree=4, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=- 1)
regressor.fit(X,y)

#predicting values for 6.5 years of experience

y_pred = regressor.predict([[6.5]])

#bydefault or Parameter tunning we got $130,001 for 6.5 years of experience but actual salary is 150K

plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title("Truth or Bluff('SVR')")
plt.xlabel('Position Salary')
plt.ylabel('Experience')
plt.show()

#after using hyper parameter tunning we got accuracy of 175,707