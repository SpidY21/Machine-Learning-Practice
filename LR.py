import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"C:\Users\Yash\Desktop\EXCELR\ML Udemy\Salary_Data.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rg = LinearRegression()
rg.fit(X_train, y_train)

y_pred = rg.predict(X_test)

# plt.scatter(X_train, y_train, color='r')
# plt.plot(X_train, rg.predict(X_train), color='b')
# plt.xlabel('Experience')
# plt.ylabel('Salary')
# plt.title('Salary vs Experience')
# plt.show()

# plt.scatter(X_test, y_test, color='r')
# plt.plot(X_test, rg.predict(X_test), color='b')
# plt.xlabel('Experience')
# plt.ylabel('Salary')
# plt.title('Predicted Salary vs Experience')
# plt.show()
print(rg.predict([[12]]))