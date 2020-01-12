import pandas as np

data = np.read_csv("./data/autompg.csv")
features = data.columns.values[1:][:-1]
print(features)

x = data[features].values
y = data['mpg']

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=3)
regressor.fit(x, y)
print(regressor.fit(x, y))

from sklearn.tree import export_graphviz
export_graphviz(regressor, out_file='tree_multi.dot', feature_names=features)

from subprocess import call
call("dot -Tpng tree_multi.dot -o tree_multi.png")

from PIL import Image
Image.open('./tree_multi.png').show()

from sklearn.metrics import  mean_absolute_error
from sklearn.metrics import mean_squared_error
print(mean_absolute_error(y, regressor.predict(x))) #loi tri tuyet doi
print(mean_squared_error(y, regressor.predict(x))) #loi binh phuong

