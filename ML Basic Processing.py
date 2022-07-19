#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv(r"G:\Data Science\Videos\Material\Mach Notes\PART-1- CLASS WORK\Data.csv")

X = dataset.iloc[:,:-1].values #independent variable
y=dataset.iloc[:,3].values #dependent variables, we have to use ".values" becasue it will create a seperate array else it will be dataframe

from sklearn.impute import SimpleImputer #imputation technique.

#now in our dataset we have missing values, we will fix this numerical values with mean strategy. if we have to use mode then we have to use work 'most_frequent' in place of mean

imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
#mean strategy Age = 38.77, Salary = 63,777.777
#median strategy Age = 38, Salary = 61000
#mode Strategy Age = 27, Salary = 48000
#now applying imputation technique

imputer  = imputer.fit(X[:,1:3]) # we have numerical values in first and second column so we have choosed 1:3 columns

X[:,1:3]= imputer.transform(X[:,1:3]) # till last query imputer identifier was holding the data, now we are replacing X independent variable data with imputation technique. 
#not the missing values will be filled with mean strategy, this we have applied only on numerical column

#Great we have successfully replaced missing data with mean strategy
#now its time to learn how to encode categorical data and create dummy variables

from sklearn.preprocessing import LabelEncoder #LabelEncoder is the module to appply label encoding techniques on categorical data

labelencoder_X = LabelEncoder() # identifier is hoding the module

labelencoder_X.fit_transform(X[:,0])

X[:,0]= labelencoder_X.fit_transform(X[:,0]) # by using label encoder we have repalced France-0, Germany with -1 and Spain -2

labelencoder_y=LabelEncoder()

y = labelencoder_y.fit_transform(y) #becasue machine does not understand Yes and No so we have to conver it into numerical. As this is a text field we change to numeric by using label encoder.

#Splitting the dataset in training and testing set 

from sklearn.model_selection import train_test_split #impting module train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0, test_size=0.2)
#if you remove random_state then your model will not behave as accurate

# we requried feature scalling here because age numer and salary numers are not equal and machin will conside salary as greater than age.
# we have two scaling techniques one is standardscaler and second one is normalizer.
# **Feature Scaling**

#standardscaler
#from sklearn.preprocessing import standardscaler
#sc_X  = standardscaler() #variable holds the module

#Normalizer
from sklearn.preprocessing import Normalizer
sc_X  = Normalizer() #variable holds the module

X_train = sc_X.fit_transform(X_train)

x_test = sc_X.fit_transform(X_test)


