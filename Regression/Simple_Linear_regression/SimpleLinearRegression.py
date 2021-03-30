#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Importing Dataset
dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
# Splitting dataset into train and test set
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

#Training Simple Linear Regression model on the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set  

y_pred = regressor.predict(X_test)

# Visualizing the Training set results

plt.scatter(X_train,y_train,color='Red')
plt.plot(X_train,regressor.predict(X_train),color ='Blue')
plt.title("Salary Vs Experience(Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualizing the Test set results

plt.scatter(X_test,y_test,color='Red')
plt.plot(X_train,regressor.predict(X_train),color='Blue')
plt.title("Salary Vs Experience(Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
