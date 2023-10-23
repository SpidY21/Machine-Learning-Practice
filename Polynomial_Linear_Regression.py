import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv(r"C:\Users\Yash\Desktop\EXCELR\ML Udemy\Position_Salaries.csv")

X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

nlr = LinearRegression()
nlr.fit(X, y)

plt.scatter(X, y, color='red')
plt.plot(X, nlr.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

plr = PolynomialFeatures(degree=5)
X_poly = plr.fit_transform(X)
plr2 = LinearRegression()
plr2.fit(X_poly, y)

plt.scatter(X, y, color='red')
plt.plot(X, plr2.predict(plr.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X.values), max(X.values) + 0.1, 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, plr2.predict(plr.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print(nlr.predict([[6.5]]), plr2.predict(plr.fit_transform([[6.5]])))
