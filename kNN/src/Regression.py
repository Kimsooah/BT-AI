print("Boston housing data")
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np

### Boston housing data
# get data
boston = pd.read_csv('data/Boston.csv')
print(boston.head())
boston = boston.drop('Unnamed: 0', axis=1)
print(boston.head())

y = boston['medv'].values
x = boston.drop('medv', axis=1).values

# Dự báo giá nhà dựa vào một biến
# use data 1 var
xrm = x[:, 5]
xrm = xrm.reshape(-1, 1)
y = y.reshape(-1, 1)

import matplotlib.pyplot as plt
plt.scatter(xrm, y)
plt.ylabel('y: Value of house / 1000 USD')
plt.xlabel('x: Number of rooms')
plt.show()

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(xrm, y)
print(reg.score(xrm, y))

xx = np.linspace(min(xrm), max(xrm)).reshape(-1, 1)
plt.scatter(xrm, y, color="blue")
plt.plot(xx, reg.predict(xx), color="red", linewidth=3)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()

from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(reg, hist=False)
visualizer.fit(xrm, y)
visualizer.score(xrm, y)
visualizer.poof()

# use data multi var
# split data: 70%-training 30%-testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print("R^2 = ", reg.score(x_train, y_train))

from yellowbrick.regressor import ResidualsPlot
viz = ResidualsPlot(reg, hist=False)
viz.fit(x_train, y_train)
viz.score(x_test, y_test)
viz.poof();

# Ridge Regression
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(x_train, y_train)
print("R^2 = ", ridge.score(x_train, y_train))

ridge_pred = ridge.predict(x_test)
print(ridge.score(x_test, y_test))

from sklearn.linear_model import RidgeCV
from yellowbrick.regressor import AlphaSelection
alphas = np.logspace(-10, 1, 400)
model = RidgeCV(alphas=alphas)
visualizer = AlphaSelection(model)

y_train = y_train.ravel()
print(y_train.shape)
visualizer.fit(x_train, y_train)
visualizer.poof()

# Lasso regression
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(x_train, y_train)
lasso_pred = lasso.predict(x_test)
print("R^2 = ", lasso.score(x_test, y_test))

from sklearn.linear_model import LassoCV
from yellowbrick.regressor import AlphaSelection
alphas = np.logspace(-10, 1, 400)
model = LassoCV(alphas=alphas)
visualizer = AlphaSelection(model)

y_train = y_train.ravel()
visualizer.fit(x_train, y_train)
g = visualizer.poof()

# Lasso regression for feature selection
names = boston.columns.values
print(names)
names = names[:13]
print(names)

lasso_coef = lasso.fit(x, y).coef_
print(len(lasso_coef))
print(len(names))

plt.plot(range(len(names)), lasso_coef)
plt.xticks(range(len(names)), names, rotation=60)
plt.ylabel('Coefficients')
plt.show()

plt.scatter(xrm, y)
plt.xlabel('x: Numberof rooms')
plt.ylabel('y: Values of house / 1000 USD')
plt.show()

# use KNN
from sklearn.neighbors import KNeighborsRegressor
reg = KNeighborsRegressor(n_neighbors=1)
reg.fit(xrm, y)

xx = np.linspace(min(xrm), max(xrm)).reshape(-1, 1)
plt.scatter(xrm, y, color='blue')
plt.plot(xx, reg.predict(xx), color='red', linewidth=3)
plt.ylabel('y: Value of house / 1000 USD')
plt.xlabel('x: Number of rooms')
plt.show()

print("Test set R^2: {:.2f}".format(reg.score(xrm, y)))
reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(xrm, y)

xx=np.linspace(min(xrm),max(xrm)).reshape(-1,1)
plt.scatter(xrm,y,color="blue")
plt.plot(xx,reg.predict(xx),color="red",linewidth=3)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()

# selection k good
from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}
reg = KNeighborsRegressor()
model = GridSearchCV(reg, params, cv=5)
model.fit(xrm,y)
print(model.best_params_)

reg = KNeighborsRegressor(n_neighbors=9)
reg.fit(xrm, y)

xx = np.linspace(min(xrm),max(xrm)).reshape(-1,1)
plt.scatter(xrm,y,color="blue")
plt.plot(xx,reg.predict(xx),color="red",linewidth=3)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()

# Dự báo giá nhà với tất cả các biến
reg = KNeighborsRegressor(n_neighbors = 3)
reg.fit(x_train, y_train)
reg_pred = reg.predict(x_test)
print(reg.score(x_test, y_test))
