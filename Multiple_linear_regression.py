import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

dataset = pd.read_csv(r"C:\Users\Yash\Desktop\EXCELR\ML Udemy\50_Startups.csv")

df = pd.get_dummies(dataset.State)
dataset = pd.concat([df, dataset], axis=1)
dataset = dataset.drop('State', axis=1)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
rg = LinearRegression()
rg.fit(X_train, y_train)

plt.scatter(y_test, rg.predict(X_test), color='r')
# plt.plot(X_test, rg.predict(X_test), color='b')
plt.show()
