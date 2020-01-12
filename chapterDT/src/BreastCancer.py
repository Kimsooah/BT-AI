from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

cancer = load_breast_cancer()
print(cancer.keys())
print(cancer.data.shape)
print(cancer.feature_names)
print(cancer.target_names)

x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
print(x_test.shape)
print(x_train.shape)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(x_train, y_train)
y_pred = tree.predict(x_test)
print('Train: %.4f\nTest: %.4f'%(tree.score(x_train, y_train), tree.score(x_test, y_test)))

export_graphviz(tree, out_file='tree_classifier.dot', feature_names=cancer.feature_names,
                class_names=cancer.target_names, impurity=False, filled=True)

from subprocess import call
call('dot.exe -Tpng tree_classifier.dot -o tree_classifier.png')

from PIL import Image
Image.open("tree_classifier.png").show()

print(tree.feature_importances_)

import matplotlib.pyplot as plt
features = cancer.feature_names
n = len(features)
plt.figure(figsize=(8,10))
plt.barh(range(n),tree.feature_importances_)
plt.yticks(range(n),features)
plt.title('Muc do quan trong cac thuoc tinh')
plt.ylabel('Cac thuoc tinh')
plt.xlabel('Muc do')
plt.show()

