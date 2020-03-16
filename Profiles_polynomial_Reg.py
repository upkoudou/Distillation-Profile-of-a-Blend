
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Crude-File.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, 1:].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Polynomial Regression to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Model score
from sklearn.metrics import mean_squared_error, r2_score

#train
y_poly_pred_train_ = lin_reg_2.predict(poly_reg.fit_transform(X_train))

train_rmse = np.sqrt(mean_squared_error(y_train,y_poly_pred_train_))
train_r2 = r2_score(y_train,y_poly_pred_train_)
print(train_rmse)
print(train_r2)

#test
y_poly_pred_test_ = lin_reg_2.predict(poly_reg.fit_transform(X_test))

test_rmse = np.sqrt(mean_squared_error(y_test,y_poly_pred_test_))
test_r2 = r2_score(y_test,y_poly_pred_test_)
print(test_rmse)
print(test_r2)
