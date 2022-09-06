#Cross validation KFold and Grid Search
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"E:\Data Science Post Class\Machine Learning\Classification\Cross Validation and Model Tunning\1.K-FOLD CROSS VALIDATION CODE_ MODEL SELECTION\Social_Network_Ads.csv")

X = dataset.iloc[:,2:-1].values
y=dataset.iloc[:,-1].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0, test_size =0.25)

#importing SVC model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train,y_train)

#Predicting values
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

#Accuracy
from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test, y_pred)
ac

bias = classifier.score(X_train,y_train)
bias

variance = classifier.score(X_test,y_test)
variance

#applying K-Fold Cross Validations
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X_train,y=y_train, cv=10)
print("Best Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

#applying Grid search to find the best model and the best parameter
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator= classifier, 
                           param_grid= parameters,
                           scoring = 'accuracy',
                           cv= 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameter = grid_search.best_params_
print("Best Accuracy: {:.2f}".format(best_accuracy*100))
print("Best Parameter: ",best_parameter)








