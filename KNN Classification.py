#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"E:\Data Science Post Class\Machine Learning\Classification\Logistic Classification\2.LOGISTIC REGRESSION CODE\Social_Network_Ads.csv")

X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1:].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler  # Normalizer
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#KNN Classification 
from sklearn.neighbors import KNeighborsClassifier
KNN_classifier = KNeighborsClassifier()
KNN_classifier.fit(X_train,y_train.ravel())

y_KNN = KNN_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_KNN)
cm

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_KNN)
ac

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_KNN)
cr


bias = KNN_classifier.score(X_train,y_train)
bias

variance = KNN_classifier.score(X_test,y_test)
variance

fut_data = pd.read_excel(r"E:\Data Science Post Class\Machine Learning\Classification\future prediction.xlsx")
fut_X = fut_data.iloc[:,[2,3]].values
fut_X = sc.fit_transform(fut_X)
KNN_prediction = KNN_classifier.predict(fut_X)

future_prediction = pd.read_excel(r"E:\Data Science Post Class\Machine Learning\Classification\predicted1.xlsx")
future_prediction["KNN Prediction"]=KNN_prediction

future_prediction.to_excel("E:\Data Science Post Class\Machine Learning\Classification\predicted1.xlsx",index=False)




