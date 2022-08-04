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

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l1',solver='liblinear',max_iter=200,multi_class='ovr')
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
ac

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
cr


bias = classifier.score(X_train,y_train)
bias

variance = classifier.score(X_test,y_test)
variance


# Standard Scaler and 25% --> accuracy = 89 Bias = 82.3% and Varaince = 89% This is good
# normalizer and 25% --> accuracy = 68 Bias = 63% and Varaince = 68%

# Standard Scaler and 30% --> accuracy = 86.6 Bias = 83.2% and Varaince = 86% This is good
# normalizer and 30% --> accuracy = 65.8 Bias = 63.5% and Varaince = 65.8%

# Standard Scaler and 20% --> accuracy = 92.5 Bias = 82.1% and Varaince = 92.5% This is the best model
# normalizer and 20% --> accuracy = 72.5 Bias = 62.1% and Varaince = 72.5%

# from above 3 test set one thing is confirm we got good results with Standard Scaler, with test sample 20% we got best model with parameter tunning.

fut_data = pd.read_excel(r"E:\Data Science Post Class\Machine Learning\Classification\Logistic Classification\future prediction.xlsx")

fu_data = fut_data.iloc[:,[2,3]].values
fu_data = sc.fit_transform(fu_data)

LogisticRegressin_pred=classifier.predict(fu_data)

fut_data['LogisticRegression'] = LogisticRegressin_pred

#saving file
np.savetxt("E:\Data Science Post Class\Machine Learning\Classification\Logistic Classification\predicted.xlsx",fut_data,fmt='%s')

predicted = pd.DataFrame(fut_data)
predicted.to_excel("E:\Data Science Post Class\Machine Learning\Classification\Logistic Classification\predicted.xlsx",index=False)
