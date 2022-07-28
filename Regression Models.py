import numpy as np
import pandas as pd

dataset=pd.read_csv(r"E:\Data Science Post Class\Machine Learning\Regression\Position_Salaries.csv")
dataset = dataset.iloc[:-1,:] # as we assumed that CEO salary is outlier we removed it from our dataset

X = dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


#simple Linear Regression
from sklearn.linear_model  import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(X,y)

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 13) #by default it will be two degree, default parameter is called parameter tunning and when we change the degree it is called hyper parametr tunning
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly,y)

#second linear regression with X_poly
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#predicting the salary
#lin_reg.predict([[6.5]]) #predicting salary with linear model

poly_predict = lin_reg_2.predict(poly_reg.fit_transform([[6.5]])) #predicting salary with poly model

#with degree 13 we got polynomila regression prediction of 175251

#SVR
from sklearn.svm import SVR
SVR_regressor = SVR(kernel='poly', degree=4, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=- 1)
SVR_regressor.fit(X,y)
SVR_predict = SVR_regressor.predict([[6.5]])
# SVR Predicts 175708 for 6.5 years of experience

# Decision Tree predicts 175K 100% accurate results we got
from sklearn.tree import DecisionTreeRegressor
Decision_reg = DecisionTreeRegressor(criterion='poisson', splitter='random', max_depth=None, min_samples_split=0.3, min_samples_leaf=0.1, min_weight_fraction_leaf=0.2, max_features='auto', random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0)
Decision_reg.fit(X,y)

Decision_predict = Decision_reg.predict([[6.5]])

from sklearn.ensemble import RandomForestRegressor
RandomForest_reg = RandomForestRegressor(n_estimators=100,random_state=0,criterion='poisson',min_samples_leaf=2)
RandomForest_reg.fit(X,y)

RandomForest_predict = RandomForest_reg.predict([[6.5]])
# Random Forest predicts for 6.5 years of experience is $175,770

from sklearn.neighbors import KNeighborsRegressor
Knn_reg = KNeighborsRegressor(n_neighbors=2, weights='uniform', algorithm='kd_tree', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
Knn_reg.fit(X,y)

Knn_predict = Knn_reg.predict([[6.5]])

#Knn regression model predicts exact value of $175,000 for 6.5 years of experience

